#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

rm -rf build dist *.egg-info src/*.egg-info src/leenef.egg-info
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*

echo "Built and checked:"
ls -lh dist/
