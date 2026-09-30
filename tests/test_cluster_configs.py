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
from pathlib import Path
from typing import Any

import pytest
import yaml
from kneeno.evaluation import load_eval_config

from lightly_train._methods.dinov2.dinov2 import DINOv2Args
from lightly_train._methods.dinov2.dinov2_transform import DINOv2ViTTransformArgs

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
