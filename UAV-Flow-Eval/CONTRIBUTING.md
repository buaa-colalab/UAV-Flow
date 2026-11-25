# Contributing

Use a fresh Python 3.11 environment: [installation](README.md#0-installation).
Keep running deployment environments unchanged.

## Checks

```bash
ruff format --check .
ruff check .
mypy --no-incremental
pytest -m 'not simulator'
python -m build
```

- Runtime dependencies: `requirements.txt`; extras and type-check scope: `pyproject.toml`.
- Code/data: `src/`; tests and frozen fixtures: `tests/`.
- Style: 100 columns, double quotes, LF. Preserve registration imports and public re-exports.
- Preserve training and evaluation behavior; review protocol changes separately.
- Default tests are CPU-only. Real UE tests require a disposable instance; never target active training services.
- Retain [LICENSE](LICENSE) and record provenance for new code/data; model weights and UE assets stay external.
- Project-authored code uses Apache-2.0; project-authored datasets, images and documentation use [CC BY-NC 4.0](LICENSE-CONTENT). Preserve third-party terms.

## Test the built wheel

Install into another fresh environment and run outside the checkout:

```bash
python3.11 -m venv /tmp/uav-flow-wheel
/tmp/uav-flow-wheel/bin/python -m pip install 'pip==24.0' 'setuptools==79.0.1' 'wheel==0.48.0'
/tmp/uav-flow-wheel/bin/python -m pip install dist/*.whl
(cd /tmp && /tmp/uav-flow-wheel/bin/python -I -B -m uav_flow --help)
```

Before publishing, verify task manifests, license notices and distribution contents.
