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
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from kneeno.dataset import split_hash
from kneeno.evaluation import load_eval_config
from kneeno.evaluation.config import ALL_TASKS

import lightly_train
from lightly_train._callbacks.callback_args import CallbackArgs
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
DATA = {"out": "/out", "data_root": "/data", "data_meta": "/meta.json"}


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
        {
            "out": config["data"]["out"],
            "data": [config["data"]["data_root"]],
            **config.get("train", {}),
            "model": config["model"],
        }
    )


@pytest.mark.parametrize("path", RUN_CONFIGS, ids=lambda p: p.name)
def test_run_config__callbacks_valid(path: Path) -> None:
    # Like the train block's values, checked by pretrain only once the job runs.
    CallbackArgs.model_validate(load_run_config(path).get("callbacks", {}))


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
        "data": {
            "out": "/runs/${RUN}",
            "data_root": "$TMP/internal",
            "data_meta": "/m",
        },
        "eval": {"data": {"data_root": {"internal": "$TMP/labeled"}}},
    }
    loaded = load_run_config(_write(tmp_path, config))
    assert loaded["data"]["out"] == "/runs/vitb14"
    assert loaded["data"]["data_root"] == "/scratch/internal"
    assert loaded["eval"]["data"]["data_root"] == {"internal": "/scratch/labeled"}


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
    # The image folder of the stock pretrain argument is data_root + data_meta here.
    path = _write(tmp_path, {"model": MODEL, "data": {**DATA, "data": "/data"}})
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


# The example in the shipped configs and the script's docstring.
CALLBACKS = {"model_checkpoint": {"every_n_epochs": 50, "save_top_k": -1}}


def test_load_run_config__callbacks_block(tmp_path: Path) -> None:
    callbacks = {**CALLBACKS, "early_stopping": None}
    path = _write(tmp_path, {"model": MODEL, "data": DATA, "callbacks": callbacks})
    # A null callback switches it off, so it must survive loading.
    assert load_run_config(path)["callbacks"] == callbacks
    CallbackArgs.model_validate(callbacks)


def test_load_run_config__callbacks_kneeno_eval(tmp_path: Path) -> None:
    callbacks = {"kneeno_eval": {"config": {"freq": {"knn": 1}}}}
    path = _write(tmp_path, {"model": MODEL, "data": DATA, "callbacks": callbacks})
    with pytest.raises(ValueError, match="top-level 'eval' block"):
        load_run_config(path)


# Case id -> global split of the fake metadata _data_block writes.
SPLITS = {"c3": "train", "c1": "train", "c2": "unlabeled", "c0": "val"}


def _data_block(tmp_path: Path, splits: dict[str, str] = SPLITS) -> dict[str, str]:
    """A data block whose data_root has a folder for every case of ``splits``, one
    folder in no split (``c9``), and whose data_meta groups the cases by ``splits``."""
    root = tmp_path / "internal"
    metadata: dict[str, dict[str, Any]] = {}
    for case_id, split in splits.items():
        (root / case_id / "sag").mkdir(parents=True, exist_ok=True)
        metadata.setdefault(split, {})[case_id] = {"sag": {"dimensions": [8, 8, 3]}}
    (root / "c9").mkdir(exist_ok=True)
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps(metadata))
    return {"out": "/out", "data_root": str(root), "data_meta": str(meta)}


def _run_main(
    monkeypatch: pytest.MonkeyPatch, config: dict[str, Any], tmp_path: Path
) -> dict[str, Any]:
    """Runs cluster/pretrain_dinov2_kneeno.py's main() and returns pretrain's kwargs.

    ``config["data"]`` is replaced by ``_data_block``'s, unless the config has one
    with real files."""
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
    if config["data"] == DATA:
        config = {**config, "data": _data_block(tmp_path)}
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
    assert kwargs["out"] == "/out"
    assert "data_root" not in kwargs and "data_meta" not in kwargs
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


def test_main__passes_callbacks_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eval_block = {"freq": {"knn": 1}}
    config = {"model": MODEL, "data": DATA, "callbacks": CALLBACKS}
    kwargs = _run_main(monkeypatch, config, tmp_path)
    assert kwargs["callbacks"] == CALLBACKS
    kwargs = _run_main(monkeypatch, {**config, "eval": eval_block}, tmp_path)
    assert kwargs["callbacks"] == {
        **CALLBACKS,
        "kneeno_eval": {"config": eval_block},
    }


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


def test_main__pretrains_only_on_train_and_unlabeled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The val case c0 is the eval's internal test set, and c9 is in no split.
    kwargs = _run_main(monkeypatch, {"model": MODEL, "data": DATA}, tmp_path)
    root = tmp_path / "internal"
    assert kwargs["data"] == [str(root / "c1"), str(root / "c2"), str(root / "c3")]
    out = capsys.readouterr().out
    assert "Pretraining on 3 exams" in out
    assert "2 exam folders there are held out" in out
    assert f"sha256 of the sorted case ids: {split_hash(['c1', 'c2', 'c3'])}." in out


def test_main__no_exam_to_pretrain_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _data_block(tmp_path, splits={"c0": "val"})
    with pytest.raises(ValueError, match="no exams to pretrain on"):
        _run_main(monkeypatch, {"model": MODEL, "data": data}, tmp_path)


def _pretrain_config_of(eval_path: Path) -> Path:
    """The run config whose run an eval config evaluates: eval-<name> -> pretrain-<name>."""
    path = eval_path.with_name(eval_path.name.replace("eval-", "pretrain-", 1))
    assert path.is_file(), f"No pretraining config '{path.name}' for '{eval_path.name}'"
    return path


@pytest.mark.parametrize("path", EVAL_CONFIGS, ids=lambda p: p.name)
def test_eval_config__valid(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRETRAIN_JOB_ID", "1234")
    monkeypatch.setenv("TMP", "/tmp/job")
    raw, config = _load_eval_run_config(path)
    eval_config = load_eval_config(config.eval)
    assert "1234" in str(config.checkpoint)
    # Standalone, KneeNo's own SummaryWriter logs the test metrics, and per_label_dir
    # decides where results.json and params.yaml go.
    assert eval_config["logging"]["tensorboard_dir"] is not None
    assert eval_config["logging"]["per_label_dir"] is not None
    # Only known tasks; a task left out keeps KneeNo's default frequency, null (off).
    assert set(raw["eval"]["freq"]) <= set(ALL_TASKS)


@pytest.mark.parametrize("path", EVAL_CONFIGS, ids=lambda p: p.name)
def test_eval_config__matches_pretrain_config(path: Path) -> None:
    """The eval config evaluates its pretraining run as the in-training eval did, apart
    from the deliberate differences below. Compared after merging over KneeNo's
    defaults, so a key spelled out at its default in only one file is no difference."""
    eval_config = yaml.safe_load(path.read_text())
    run = yaml.safe_load(_pretrain_config_of(path).read_text())

    out = run["data"]["out"].replace("${SLURM_JOB_ID}", "${PRETRAIN_JOB_ID}")
    assert eval_config["checkpoint"] == f"{out}/checkpoints/last.ckpt"
    for key in ("tensorboard_dir", "per_label_dir"):
        assert eval_config["eval"]["logging"][key].startswith(f"{out}/eval/")
    assert eval_config["image_size"] == run["transform"]["image_size"]

    run_eval = load_eval_config(run["eval"])
    # The head settings of a task the pretraining run never runs are not comparable.
    in_training = {task for task, freq in run_eval["freq"].items() if freq}

    def without_differences(block: dict[str, Any]) -> dict[str, Any]:
        # Where it logs, which tasks run, and the data setup of a standalone job: all
        # exams, the external test set too (its paths included), extracted to $TMP
        # instead of /dev/shm, more workers, and a broken exam raises instead of being
        # replaced (the in-training eval must not kill the run). The standalone heads
        # also train longer (max_epochs, warmup_epochs): they are trained once, not
        # every few pretraining epochs.
        block = {
            k: v
            for k, v in block.items()
            if k not in ("logging", "freq")
            and not (k in ALL_TASKS and k not in in_training)
        }
        for task in in_training:
            block[task] = {
                k: v
                for k, v in block[task].items()
                if k not in ("max_epochs", "warmup_epochs")
            }
        block["data"] = {
            k: v
            for k, v in block["data"].items()
            if k
            not in (
                "subset",
                "data_root",
                "label_meta",
                "test_sets",
                "num_workers",
                "abort_on_io_error",
            )
        }
        return block

    assert without_differences(load_eval_config(eval_config["eval"])) == (
        without_differences(run_eval)
    )
    # The same internal data; only the standalone eval tests on the external dataset too.
    eval_data, run_data = eval_config["eval"]["data"], run["eval"]["data"]
    assert eval_data["label_meta"]["internal"] == run_data["label_meta"]["internal"]
    assert run_data["test_sets"] == ["internal"]
    assert eval_data["test_sets"] == ["internal", "external"]


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
