#!/usr/bin/env bash
# Resolve an approved immutable target, hard-sync to it, refresh dependencies,
# restart, health-check, and roll back on failure.
# Run by roadresilience-update.timer (user systemd) every ~2 min. No sudo needed.
set -euo pipefail

REPO="$HOME/Trace"
cd "$REPO"

# Public production must opt into an explicit release ref. The service reads it
# from ~/.config/roadresilience/deploy.env; refusing an empty value prevents an
# accidental return to continuously shipping the checked-out dev branch.
DEPLOY_REF="${DEPLOY_REF:-}"
if [[ "$DEPLOY_REF" =~ ^[0-9a-fA-F]{40}$ ]]; then
    TARGET_REF="${DEPLOY_REF}^{commit}"
elif [[ "$DEPLOY_REF" =~ ^v[0-9]+\.[0-9]+(\.[0-9]+)?$ ]]; then
    TARGET_REF="refs/tags/${DEPLOY_REF}^{commit}"
else
    echo "DEPLOY_REF must be an immutable application release tag or full 40-character commit SHA" >&2
    exit 2
fi

git fetch --quiet --tags origin

if ! TARGET="$(git rev-parse --verify "$TARGET_REF" 2>/dev/null)"; then
    echo "DEPLOY_REF does not resolve to a commit: $DEPLOY_REF" >&2
    exit 2
fi
LOCAL="$(git rev-parse @)"

if [ "$LOCAL" = "$TARGET" ]; then
    exit 0   # already up to date
fi

echo "$(date -Is) update: ${LOCAL:0:8} -> ${TARGET:0:8}"

rollback() {
    echo "$(date -Is) ERROR: deploy transaction failed @ ${TARGET:0:8}; rolling back to ${LOCAL:0:8}" >&2
    git reset --hard --quiet "$LOCAL"
    ./.venv/bin/pip install -q -r deploy/requirements-app.txt || true
    npm ci --prefix web --no-audit --no-fund >/dev/null 2>&1 || true
    npm run build --prefix web >/dev/null 2>&1 || true
    systemctl --user restart roadresilience.service || true
}
trap rollback ERR

# Deploy checkout is read-only — hard-sync beats pull (no merge state to break).
git reset --hard --quiet "$TARGET"

# refresh deps (cheap no-op when already satisfied)
./.venv/bin/pip install -q -r deploy/requirements-app.txt
npm ci --prefix web --no-audit --no-fund
npm run build --prefix web
test -f web/out/index.html

export XDG_RUNTIME_DIR="/run/user/$(id -u)"
systemctl --user restart roadresilience.service

# Health gate: poll FastAPI (~30 s max), roll back on failure.
healthy() {
    for _ in 1 2 3 4 5 6; do
        sleep 5
        if curl -fsS http://127.0.0.1:8000/healthz >/dev/null 2>&1 \
            && curl -fsS http://127.0.0.1:8000/ >/dev/null 2>&1; then
            return 0
        fi
    done
    return 1
}

if healthy; then
    trap - ERR
    echo "$(date -Is) restarted roadresilience @ ${TARGET:0:8} (healthy)"
else
    echo "$(date -Is) ERROR: health check FAILED @ ${TARGET:0:8} — ROLLING BACK to ${LOCAL:0:8}" >&2
    git reset --hard --quiet "$LOCAL"
    ./.venv/bin/pip install -q -r deploy/requirements-app.txt
    npm ci --prefix web --no-audit --no-fund
    npm run build --prefix web
    test -f web/out/index.html
    systemctl --user restart roadresilience.service
    if healthy; then
        echo "$(date -Is) rollback to ${LOCAL:0:8} is up — investigate ${TARGET:0:8} before re-deploying" >&2
    else
        echo "$(date -Is) CRITICAL: rollback also unhealthy — manual intervention required" >&2
    fi
    trap - ERR
    exit 1
fi
