#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""The cluster run configs and ``cluster/run_config.py``'s ``load_run_config``."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from kneeno.evaluation import load_eval_config

import lightly_train
from lightly_train._commands.train import FunctionTrainConfig
from lightly_train._methods.dinov2.dinov2 import DINOv2Args
from lightly_train._methods.dinov2.dinov2_transform import DINOv2ViTTransformArgs
from lightly_train._models.dinov2_vit.dinov2_vit_package import DINOv2ViTPackage

CLUSTER_DIR = Path(__file__).parents[1] / "cluster"
RUN_CONFIGS = sorted((CLUSTER_DIR / "configs").glob("*.yaml"))


# cluster/ is not a package, so load run_config.py from its path.
_spec = importlib.util.spec_from_file_location(
    "run_config", CLUSTER_DIR / "run_config.py"
)
assert _spec is not None and _spec.loader is not None
run_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_config)
load_run_config = run_config.load_run_config


MODEL = "dinov2/_vittest14"
DATA = {
    "out": "/out",
    "data_root": "/data",
    "data_meta": "/meta.json",
    "series_depth": 24,
    "resample_mode": "nearest",
}


def _write(tmp_path: Path, config: dict[str, Any]) -> Path:
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_run_configs_exist() -> None:
    assert RUN_CONFIGS


@pytest.mark.parametrize("path", RUN_CONFIGS, ids=lambda p: p.name)
def test_run_config__valid(path: Path) -> None:
    config = load_run_config(path)
    DINOv2Args.model_validate(config.get("method", {}))
    DINOv2ViTTransformArgs.model_validate(config.get("transform", {}))
    if "eval" in config:
        eval_config = load_eval_config(config["eval"])
        # Metrics are re-logged through lightly-train's loggers, so KneeNo's own
        # SummaryWriter would double-write.
        assert eval_config["logging"]["tensorboard_dir"] is None
        # DINOv2 has a cls token, so unlike vjepa2 the linear task stays enabled.
        assert eval_config["freq"]["linear"] is not None


@pytest.mark.parametrize("path", RUN_CONFIGS, ids=lambda p: p.name)
def test_run_config__train_valid(path: Path) -> None:
    # load_run_config only checks the train keys' names; their values are checked
    # by pretrain's own config, which would otherwise only fail on the cluster.
    config = load_run_config(path)
    FunctionTrainConfig.model_validate(
        {**config["data"], **config.get("train", {}), "model": config["model"]}
    )


@pytest.mark.parametrize("path", RUN_CONFIGS, ids=lambda p: p.name)
def test_run_config__model_builds(path: Path) -> None:
    name = load_run_config(path)["model"].removeprefix("dinov2/")
    DINOv2ViTPackage.get_model(name, num_input_channels=1, load_weights=False)


@pytest.mark.parametrize("path", RUN_CONFIGS, ids=lambda p: p.name)
def test_run_config__compile_blocks_share_graphs(path: Path) -> None:
    # compile_blocks compiles every block on its own, and the blocks only share Dynamo's compiled graphs if
    # nothing they branch on differs between them. Per-block drop path rates (drop_path_uniform: false) give
    # every block its own graphs: past Dynamo's recompile limit of 8 the remaining blocks silently run eagerly.
    config = load_run_config(path)
    if not config.get("method", {}).get("compile_blocks", False):
        pytest.skip("compile_blocks is off")
    name = config["model"].removeprefix("dinov2/")
    model = DINOv2ViTPackage.get_model(name, num_input_channels=1, load_weights=False)
    assert not model.chunked_blocks
    assert len({block.sample_drop_ratio for block in model.blocks}) == 1


def test_load_run_config__missing_and_null_blocks(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        {"model": MODEL, "data": DATA, "method": {"batch_norm": True}, "eval": None},
    )
    assert load_run_config(path) == {
        "model": MODEL,
        "data": DATA,
        "method": {"batch_norm": True},
    }


def test_load_run_config__expands_env_vars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TMP", "/scratch")
    monkeypatch.setenv("RUN", "vitb14")
    config = {
        "model": MODEL,
        "data": {**DATA, "out": "/runs/${RUN}", "data_root": "$TMP/unlabeled"},
        "eval": {"data": {"data_root": "$TMP/labeled"}},
    }
    loaded = load_run_config(_write(tmp_path, config))
    assert loaded["data"]["out"] == "/runs/vitb14"
    assert loaded["data"]["data_root"] == "/scratch/unlabeled"
    assert loaded["eval"]["data"]["data_root"] == "/scratch/labeled"


def test_load_run_config__unknown_key(tmp_path: Path) -> None:
    path = _write(
        tmp_path, {"model": MODEL, "data": DATA, "methods": {"batch_norm": True}}
    )
    with pytest.raises(ValueError, match="unknown top-level keys"):
        load_run_config(path)


@pytest.mark.parametrize("model", [None, {"name": MODEL}])
def test_load_run_config__model_required_string(tmp_path: Path, model: Any) -> None:
    config = {"data": DATA} if model is None else {"model": model, "data": DATA}
    with pytest.raises(ValueError, match="needs 'model' as a string"):
        load_run_config(_write(tmp_path, config))


def test_load_run_config__no_data_block(tmp_path: Path) -> None:
    path = _write(tmp_path, {"model": MODEL, "method": {"batch_norm": True}})
    with pytest.raises(ValueError, match="no 'data' block"):
        load_run_config(path)


@pytest.mark.parametrize("key", list(DATA))
def test_load_run_config__data_key_required(tmp_path: Path, key: str) -> None:
    data = {k: v for k, v in DATA.items() if k != key}
    path = _write(tmp_path, {"model": MODEL, "data": data})
    with pytest.raises(ValueError, match=f"missing required keys \\['{key}'\\]"):
        load_run_config(path)


def test_load_run_config__unknown_data_key(tmp_path: Path) -> None:
    path = _write(tmp_path, {"model": MODEL, "data": {**DATA, "data_rot": "/data"}})
    with pytest.raises(ValueError, match="unknown keys"):
        load_run_config(path)


def test_load_run_config__eval_depth_mismatch(tmp_path: Path) -> None:
    config = {"model": MODEL, "data": DATA, "eval": {"data": {"series_depth": 16}}}
    with pytest.raises(ValueError, match="eval.data.series_depth"):
        load_run_config(_write(tmp_path, config))


def test_train_keys_are_pretrain_arguments() -> None:
    assert set(run_config.TRAIN_KEYS) <= set(
        inspect.signature(lightly_train.pretrain).parameters
    )
    # Passed to pretrain together with the data block, so they must not overlap.
    assert not set(run_config.TRAIN_KEYS) & set(run_config.DATA_KEYS)


def test_load_run_config__train_block(tmp_path: Path) -> None:
    train = {"batch_size": 32, "epochs": 2, "trainer_args": {"limit_train_batches": 5}}
    path = _write(tmp_path, {"model": MODEL, "data": DATA, "train": train})
    assert load_run_config(path)["train"] == train


@pytest.mark.parametrize("key", ["batch_sise", "strategy", "out"])
def test_load_run_config__unknown_train_key(tmp_path: Path, key: str) -> None:
    path = _write(tmp_path, {"model": MODEL, "data": DATA, "train": {key: 1}})
    with pytest.raises(ValueError, match="'train' in run config .* unknown keys"):
        load_run_config(path)


def _run_main(
    monkeypatch: pytest.MonkeyPatch, config: dict[str, Any], tmp_path: Path
) -> dict[str, Any]:
    """Runs cluster/pretrain_dinov2_kneeno.py's main() and returns pretrain's kwargs."""
    # Restores the env var the script sets at import time.
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    # The script imports run_config by name, as if run from cluster/.
    monkeypatch.setitem(sys.modules, "run_config", run_config)
    spec = importlib.util.spec_from_file_location(
        "pretrain_dinov2_kneeno", CLUSTER_DIR / "pretrain_dinov2_kneeno.py"
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(lightly_train, "pretrain", lambda **kw: calls.append(kw))
    monkeypatch.setenv("SLURM_NNODES", "2")
    monkeypatch.delenv("SLURM_NTASKS", raising=False)
    path = _write(tmp_path, config)
    monkeypatch.setattr(
        sys, "argv", ["pretrain_dinov2_kneeno.py", "--config", str(path)]
    )
    script.main()
    assert len(calls) == 1
    return calls[0]


def test_main__passes_train_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = {"batch_size": 32, "epochs": 2, "trainer_args": {"limit_train_batches": 5}}
    kwargs = _run_main(
        monkeypatch, {"model": MODEL, "data": DATA, "train": train}, tmp_path
    )
    assert {k: kwargs[k] for k in DATA} == DATA
    assert {k: kwargs[k] for k in train} == train
    assert kwargs["model"] == MODEL
    assert kwargs["num_nodes"] == 2
    assert kwargs["callbacks"] is None


def test_main__no_train_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    kwargs = _run_main(monkeypatch, {"model": MODEL, "data": DATA}, tmp_path)
    # Left to pretrain's defaults.
    assert not set(run_config.TRAIN_KEYS) & set(kwargs)


def test_main__every_train_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A train key that the script also passes itself would raise "got multiple values
    # for keyword argument" only once the job starts on the cluster.
    train = {key: None for key in run_config.TRAIN_KEYS}
    kwargs = _run_main(
        monkeypatch, {"model": MODEL, "data": DATA, "train": train}, tmp_path
    )
    assert set(run_config.TRAIN_KEYS) <= set(kwargs)
