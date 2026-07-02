#!/usr/bin/env bash
# Auto-deploy: if the tracked branch on origin moved, pull it, refresh deps, restart.
# Run by roadresilience-update.timer (user systemd) every ~2 min. No sudo needed.
set -euo pipefail

REPO="$HOME/Trace"
cd "$REPO"

git fetch --quiet origin
LOCAL="$(git rev-parse @)"
REMOTE="$(git rev-parse '@{u}')"

if [ "$LOCAL" = "$REMOTE" ]; then
    exit 0   # already up to date
fi

echo "$(date -Is) update: ${LOCAL:0:8} -> ${REMOTE:0:8}"
git pull --ff-only --quiet

# refresh deps (cheap no-op when already satisfied)
./.venv/bin/pip install -q -r deploy/requirements-app.txt

export XDG_RUNTIME_DIR="/run/user/$(id -u)"
systemctl --user restart roadresilience.service
echo "$(date -Is) restarted roadresilience @ ${REMOTE:0:8}"
