# Repository Guidelines

## Project Structure & Module Organization
- `src/liger_kernel/`: Core package. Key submodules: `ops/`, `chunked_loss/`, `transformers/`, `triton/`, and `utils.py`.
- `test/`: PyTest suite. Fast unit tests under `test/` and longer convergence suites under `test/convergence/{fp32,bf16}`.
- `examples/`, `benchmark/`, `docs/`, `dev/`: Usage demos, perf scripts, MkDocs site, and misc dev helpers.

## Build, Test, and Development Commands
- Setup: `pip install -e .` then `pip install -e ".[dev]"`.
- Lint/format: `make checkstyle` (ruff check + format; fails on style violations).
- Unit tests: `make test` (skips convergence). Example: `pytest -q test/transformers/test_rope.py -k rope`.
- Convergence tests: `make test-convergence` (heavy; sets `HF_DATASETS_OFFLINE=1`).
- Docs: `make serve` (live preview) and `make build` (static site to `site/`).

## Coding Style & Naming Conventions
- Language: Python 3.10 target; 120‑char lines; spaces; double quotes (see `pyproject.toml`).
- Tools: ruff for linting/formatting and import order (isort rules). Run via `make checkstyle`.
- Naming: modules/functions/vars `snake_case`; classes `CamelCase`; constants `UPPER_CASE`.
- Layout: keep modules small and colocate utility code near call sites within `liger_kernel`.

## Testing Guidelines
- Framework: PyTest. Name files `test_*.py`, tests `test_*`.
- Structure: mirror `src/liger_kernel/...` under `test/...` (e.g., ops → `test/transformers/test_*` and `test/triton/*` where applicable).
- Offline by default: avoid network in tests; for convergence, respect `HF_DATASETS_OFFLINE=1`.
- Add unit tests for new logic; update/extend convergence tests when training behavior changes.

## Commit & Pull Request Guidelines
- Commits: concise, imperative titles; optionally tag scope/type (e.g., `[Fix]`, `[Feat]`), and reference issues/PRs (`#123`).
- PRs: include Summary and Testing Done; specify hardware type; run `make checkstyle`, `make test`, and (if relevant) `make test-convergence` before requesting review.

## Security & Configuration Tips
- `setup.py` auto-detects CUDA/ROCm/XPU and sets Torch/Triton deps. Ensure your env matches your hardware.
- Avoid breaking public APIs; if necessary, add deprecations and update docs/examples.

