#!/usr/bin/env bash
set -euo pipefail

python -m pytest tests_research -q
python -m pytest tests -q
python -m compileall -q src scripts external_baselines
