#!/usr/bin/env bash
set -e
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
PYTHONPATH=. python src/download_data.py
PYTHONPATH=. python src/eda.py
PYTHONPATH=. python src/train.py
PYTHONPATH=. pytest -q
echo "Setup complete. Run: PYTHONPATH=. uvicorn api.main:app --port 8000"
