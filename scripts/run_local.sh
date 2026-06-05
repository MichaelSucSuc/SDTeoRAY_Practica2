#!/bin/bash
set -e

python tests/test_workers.py
python src/train_distributed.py
