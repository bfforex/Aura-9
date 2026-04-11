#!/usr/bin/env bash
# Aura-9 Cold-Start Resumption Script
# Triggered by Windows Task Scheduler on login:
#   wsl.exe --exec /home/user/aura9/scripts/startup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Ensure infrastructure is running
echo "[Aura-9] Checking Docker services..."
docker compose -f docker/docker-compose.yaml up -d

# Wait for services to be healthy
echo "[Aura-9] Waiting for services..."
sleep 5

# Launch Aura-9 with resume
echo "[Aura-9] Starting with --resume..."
python main.py --resume
