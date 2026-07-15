# Deployment Runbook — Oracle ARM + Modal

> Repository configuration and CI are not proof of what is currently installed on the live services. Complete `Tracker.md` task O1 and record the exact deployed refs/checksums after every rollout.

Public dashboard: `https://trace.tiwaribabu.in`

## Architecture

- **Oracle Ubuntu ARM:** Streamlit, committed sample data, upload queue and CPU P2/P3.
- **Modal GPU:** authenticated P1 segmentation with the checksum-pinned v3.2 checkpoint.
- **Caddy:** public TLS/reverse proxy on 80/443.
- **Streamlit:** loopback `127.0.0.1:8501`; it must not be publicly reachable.
- **systemd user units:** application plus periodic ref synchronization; health/rollback run only when the ref changes.

`deploy/requirements-app.txt` intentionally excludes Torch/model code and includes only the dashboard plus CPU upload-analysis stack.

## Choose an approved application ref first

There is currently no GitHub **application Release** containing the July hardening work; the existing GitHub Releases are model assets, and the old `v1.0` tag predates that work. Before production rollout, approve either:

- a new immutable application tag created from reviewed `dev`, or
- a reviewed full 40-character commit SHA.

Call it `APP_REF` below. `deploy/update.sh` accepts only:

- an approved application release tag shaped `vN.N` or `vN.N.N`, resolved strictly under `refs/tags/`; or
- an exact full 40-character hexadecimal commit SHA.

It rejects moving branches such as `dev`/`main`, abbreviated SHAs, model-asset tags such as `a4-roadseg-v3.2`, malformed refs and values that do not resolve to a commit.

## One-time Oracle bring-up

Run on the host:

```bash
export APP_REF='<approved-tag-or-40-char-sha>'

sudo apt-get update
sudo apt-get install -y git curl python3.12-venv

git clone https://github.com/Akshat-Tiwari69/Trace.git ~/Trace
cd ~/Trace
git fetch --tags origin
git checkout --detach "$APP_REF"

python3.12 -m venv .venv
./.venv/bin/pip install --upgrade pip==24.2
./.venv/bin/pip install -r deploy/requirements-app.txt

mkdir -p ~/.config/roadresilience ~/.config/systemd/user
printf 'DEPLOY_REF=%s\n' "$APP_REF" > ~/.config/roadresilience/deploy.env
chmod 600 ~/.config/roadresilience/deploy.env

cp deploy/roadresilience.service ~/.config/systemd/user/
cp deploy/roadresilience-update.service ~/.config/systemd/user/
cp deploy/roadresilience-update.timer ~/.config/systemd/user/

export XDG_RUNTIME_DIR="/run/user/$(id -u)"
systemctl --user daemon-reload
systemctl --user enable --now roadresilience.service
systemctl --user enable --now roadresilience-update.timer
```

The checked-in update unit currently marks the deployment env file optional even though the script requires `DEPLOY_REF`. A45/O1 will make it fail earlier; until then, create and verify the file **before** enabling the timer.

## Application secrets

The Streamlit service needs these only for **Your imagery**:

- `MODAL_SEG_URL`
- `MODAL_SEG_KEY`

Keep them in an environment file outside the repository and owner-readable only:

```bash
chmod 600 ~/.config/roadresilience/env
```

The checked-in `roadresilience.service` does not yet load that file (tracked in A45/O1). Until the unit is fixed in an approved ref, install a user drop-in:

```bash
mkdir -p ~/.config/systemd/user/roadresilience.service.d
cat > ~/.config/systemd/user/roadresilience.service.d/env.conf <<'EOF'
[Service]
EnvironmentFile=%h/.config/roadresilience/env
EOF
systemctl --user daemon-reload
systemctl --user restart roadresilience.service
```

Sample mode works without Modal configuration.

### Rotate the shared key

1. Generate a new random key, for example `openssl rand -hex 32`.
2. Update the Modal secret `roadseg-key` (`ROADSEG_KEY=<new>`).
3. Redeploy Modal so new containers use it.
4. Update `MODAL_SEG_KEY` on the Oracle host and keep the file mode `600`.
5. Restart `roadresilience.service` and run an authenticated upload smoke.

Expect a short 401 window if Modal and Oracle are not updated atomically.

## Modal deployment

Use a separate operator environment rather than the production ARM app venv. The current local deployment tooling was verified with Modal `1.5.1`; update the pin deliberately and re-smoke before changing it.

```bash
cd ~/Trace
python3 -m venv .venv-modal
./.venv-modal/bin/pip install --upgrade pip==24.2
./.venv-modal/bin/pip install 'modal==1.5.1'
./.venv-modal/bin/modal setup
./.venv-modal/bin/modal deploy deploy/modal_app.py
```

Before deploy:

- confirm `MODEL_SHA256` in `deploy/modal_app.py` matches the intended GitHub model asset;
- verify the checkpoint loads with the pinned Modal Torch version;
- verify the Modal secret exists and is non-empty;
- record the application ref, Modal code ref and checkpoint SHA-256 in `Tracker.md`.

The endpoint checks `X-API-Key` before app-level base64 decoding/image/model work, rejects invalid/oversized payloads and returns the binary mask plus threshold metadata expected by `src/app/modal_client.py`.

## Network and Caddy

Only Caddy should be public:

- allow 80/443 in Oracle networking and host firewall;
- remove any legacy 8501 ingress/iptables rule;
- keep Streamlit bound to `127.0.0.1:8501` through `roadresilience.service`;
- install Caddy from its official Debian/Ubuntu repository (not an assumed distro package), copy `deploy/Caddyfile`, validate it, then restart Caddy;
- install the journald cap below.

```bash
# Follow the current official Debian/Ubuntu repository steps first:
# https://caddyserver.com/docs/install#debian-ubuntu-raspbian
sudo cp ~/Trace/deploy/Caddyfile /etc/caddy/Caddyfile
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl restart caddy

sudo mkdir -p /etc/systemd/journald.conf.d
sudo cp ~/Trace/deploy/journald-roadresilience.conf /etc/systemd/journald.conf.d/
sudo systemctl restart systemd-journald
```

The Caddyfile enforces the repository request-body cap. Rate limiting requires an explicit module/control decision and remains open in O1.

An external failed connection to `:8501` is encouraging but does not prove both the Oracle security list and host firewall are correct; inspect both.

## Updates and rollback

`roadresilience-update.timer` starts `roadresilience-update.service` about every two minutes. The script validates and resolves the immutable `DEPLOY_REF`, hard-resets the read-only checkout only when the target changes, refreshes dependencies, restarts Streamlit and polls its health endpoint. On failure it restores the previous commit and dependencies.

To trigger a manual update, start the service so systemd loads `DEPLOY_REF`:

```bash
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
systemctl --user start roadresilience-update.service
journalctl --user -u roadresilience-update.service -n 100 --no-pager
```

Do **not** invoke `~/Trace/deploy/update.sh` directly unless you explicitly export the same `DEPLOY_REF`; a normal shell does not load the systemd `EnvironmentFile`.

An SSH-based GitHub Action should likewise start the systemd service, not call the script without its environment.

## Health and verification

```bash
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
systemctl --user status roadresilience.service
systemctl --user status roadresilience-update.timer
journalctl --user -u roadresilience.service -n 100 --no-pager
curl -fsS http://127.0.0.1:8501/_stcore/health
curl -fsS https://trace.tiwaribabu.in/_stcore/health
```

O1 is complete only after all of these are evidenced:

1. service checkout equals the approved immutable ref;
2. public homepage and Streamlit health return 200;
3. public 8501 is closed and loopback health works;
4. sample Briefing/Analysis flow works;
5. authenticated upload covers Modal cold/warm inference, queue, CPU analysis and result;
6. an invalid key/oversized upload fails safely;
7. app restart/job recovery and rollback are exercised;
8. deployed refs/checksums/timestamp are logged in `Tracker.md`.

## Python/dependency matrix

- Development/CI: Python 3.11.
- Oracle app: isolated Python 3.12 venv.
- Modal: its pinned image/runtime and Torch version in `modal_app.py`.
- Training/local GPU: follow `SETUP.md` and the official PyTorch selector; do not reuse the Oracle app venv.

Changes to geospatial pins must pass both the full development suite and a clean `deploy/requirements-app.txt` upload-analysis smoke.
