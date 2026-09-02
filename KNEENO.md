# KneeNo integration (fork-specific)

This fork uses the sibling `../KneeNo` package for model-agnostic knee-MRI data
loading. It is **not** in `pyproject.toml` / `uv.lock` on purpose (see below) — it
is installed into the venv as a separate step.

## Setup

```bash
uv sync                      # or: uv sync --inexact  (see note)
uv pip install -e ../KneeNo  # editable; does not touch uv.lock
```

`uv pip install` bypasses project resolution, so `uv.lock` is untouched.

**Note:** a plain `uv sync` removes packages that aren't in the lock, so it will
drop `kneeno`. Either re-run the `uv pip install -e ../KneeNo` line after each
`uv sync`, or use `uv sync --inexact` (leaves extra packages in place).

## Why not a normal dependency

`kneeno` cannot be added to `[project.dependencies]` without bloating `uv.lock`
from ~30 MB to ~90 MB (GitHub's warning/limit territory). Cause: adding any new
dependency forces `uv` to re-resolve the full graph, and the current `uv`
regenerates every package's conflict-fork marker (this fork has a large
`[tool.uv] conflicts` matrix: super-gradients / onnx / onnxruntime / rfdetr plus
four torch groups) far more verbosely than the `uv` that produced the committed
lock. The re-resolve — not `kneeno` itself — is what explodes the file.

The one related change kept in `pyproject.toml` is:

```toml
[tool.uv]
exclude-newer = "2026-07-30T22:00:00Z"
```

The committed `uv.lock` was generated with this cutoff (via CLI, upstream). Pinning
it here makes `uv sync` / `uv lock --check` accept the committed lock as-is instead
of discarding it ("Ignoring existing lockfile due to removal of timestamp cutoff")
and re-resolving.
