#!/usr/bin/env bash
set -euo pipefail

if [[ -f /workspace/train.py ]]; then
    cd /workspace
else
    cd /opt/medgs
fi

exec "$@"
