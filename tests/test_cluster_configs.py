#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""The cluster run configs and ``cluster/run_config.py``'s ``load_run_config``, and the
cluster eval configs and launcher."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from kneeno.evaluation import load_eval_config
from kneeno.evaluation.config import ALL_TASKS

import lightly_train
from lightly_train._commands.eval_classification import _load_eval_run_config
from lightly_train._commands.train import FunctionTrainConfig
from lightly_train._methods.dinov2.dinov2 import DINOv2Args
from lightly_train._methods.dinov2.dinov2_transform import DINOv2ViTTransformArgs
from lightly_train._models.dinov2_vit.dinov2_vit_package import DINOv2ViTPackage

CLUSTER_DIR = Path(__file__).parents[1] / "cluster"
RUN_CONFIGS = sorted((CLUSTER_DIR / "configs").glob("pretrain-*.yaml"))
EVAL_CONFIGS = sorted((CLUSTER_DIR / "configs").glob("eval-*.yaml"))


# cluster/ is not a package, so load run_config.py from its path.
_spec = importlib.util.spec_from_file_location(
    "run_config", CLUSTER_DIR / "run_config.py"
)
assert _spec is not None and _spec.loader is not None
run_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_config)
load_run_config = run_config.load_run_config


MODEL = "dinov2/_vittest14"
DATA = {"out": "/out", "data": "/data"}


def _write(tmp_path: Path, config: dict[str, Any]) -> Path:
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_run_configs_exist() -> None:
    assert RUN_CONFIGS
    assert EVAL_CONFIGS
    # Any other file would be tested as neither.
    assert set((CLUSTER_DIR / "configs").glob("*.yaml")) == {
        *RUN_CONFIGS,
        *EVAL_CONFIGS,
    }


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
def test_run_config__stock_defaults(path: Path) -> None:
    # The 2D baseline trains with stock lightly-train: the train, method and transform
    # blocks list every key, each at its default.
    config = load_run_config(path)
    defaults = inspect.signature(lightly_train.pretrain).parameters
    assert config["train"] == {
        key: defaults[key].default for key in run_config.TRAIN_KEYS
    }
    assert config["method"] == DINOv2Args().model_dump(mode="json")
    assert config["transform"] == DINOv2ViTTransformArgs().model_dump(mode="json")


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
    DINOv2ViTPackage.get_model(name, num_input_channels=3, load_weights=False)


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
        "data": {"out": "/runs/${RUN}", "data": "$TMP/internal"},
        "eval": {"data": {"data_root": "$TMP/labeled"}},
    }
    loaded = load_run_config(_write(tmp_path, config))
    assert loaded["data"]["out"] == "/runs/vitb14"
    assert loaded["data"]["data"] == "/scratch/internal"
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
    path = _write(tmp_path, {"model": MODEL, "data": {**DATA, "data_root": "/data"}})
    with pytest.raises(ValueError, match="unknown keys"):
        load_run_config(path)


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
    assert kwargs["method"] == "dinov2"
    assert kwargs["params_file"] == str(tmp_path / "run.yaml")
    assert kwargs["resume_interrupted"] is False


def test_main__passes_eval_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eval_block = {"freq": {"knn": 1}}
    kwargs = _run_main(
        monkeypatch, {"model": MODEL, "data": DATA, "eval": eval_block}, tmp_path
    )
    assert kwargs["callbacks"] == {"kneeno_eval": {"config": eval_block}}


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


def _pretrain_config_of(eval_path: Path) -> Path:
    """The run config whose run an eval config evaluates: eval-<name> -> pretrain-<name>."""
    path = eval_path.with_name(eval_path.name.replace("eval-", "pretrain-", 1))
    assert path.is_file(), f"No pretraining config '{path.name}' for '{eval_path.name}'"
    return path


@pytest.mark.parametrize("path", EVAL_CONFIGS, ids=lambda p: p.name)
def test_eval_config__valid(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRETRAIN_JOB_ID", "1234")
    raw, config = _load_eval_run_config(path)
    eval_config = load_eval_config(config.eval)
    assert "1234" in str(config.checkpoint)
    # Standalone, KneeNo's own SummaryWriter logs the head curves, and per_label_dir
    # decides where results.json and params.yaml go.
    assert eval_config["logging"]["tensorboard_dir"] is not None
    assert eval_config["logging"]["per_label_dir"] is not None
    # Every task explicit: a missing key would keep KneeNo's default frequency.
    assert set(raw["eval"]["freq"]) == set(ALL_TASKS)


@pytest.mark.parametrize("path", EVAL_CONFIGS, ids=lambda p: p.name)
def test_eval_config__matches_pretrain_config(path: Path) -> None:
    """The eval config evaluates its pretraining run exactly as the in-training eval did,
    apart from where it logs, which tasks run and how many exams it uses."""
    eval_config = yaml.safe_load(path.read_text())
    run = yaml.safe_load(_pretrain_config_of(path).read_text())

    out = run["data"]["out"].replace("${SLURM_JOB_ID}", "${PRETRAIN_JOB_ID}")
    assert eval_config["checkpoint"] == f"{out}/checkpoints/last.ckpt"
    for key in ("tensorboard_dir", "per_label_dir"):
        assert eval_config["eval"]["logging"][key].startswith(f"{out}/eval/")
    assert eval_config["image_size"] == run["transform"]["image_size"]

    def without_differences(block: dict[str, Any]) -> dict[str, Any]:
        block = {k: v for k, v in block.items() if k not in ("logging", "freq")}
        block["data"] = {k: v for k, v in block["data"].items() if k != "subset"}
        return block

    assert without_differences(eval_config["eval"]) == without_differences(run["eval"])


def test_eval_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """cluster/eval_dinov2_kneeno.py passes the config on as eval_config."""
    # Restores the env vars the script sets at import time.
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    spec = importlib.util.spec_from_file_location(
        "eval_dinov2_kneeno", CLUSTER_DIR / "eval_dinov2_kneeno.py"
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        lightly_train, "eval_classification", lambda **kw: calls.append(kw)
    )
    path = tmp_path / "eval.yaml"
    monkeypatch.setattr(sys, "argv", ["eval_dinov2_kneeno.py", "--config", str(path)])
    script.main()
    assert calls == [{"eval_config": str(path)}]
