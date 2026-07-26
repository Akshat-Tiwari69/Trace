# Production deployment — Oracle ARM + Modal

Public site: `https://trace.tiwaribabu.in`

The production release has four deliberately small parts:

- Caddy terminates TLS on ports 80/443 and proxies to loopback.
- FastAPI serves `/api/v1/*`, the persistent CPU analysis queue, and `web/out`.
- Modal runs authenticated GPU segmentation; original uploads are not stored by the app.
- A user-systemd timer deploys only an approved immutable tag or full commit SHA and rolls back a failed health check.

There is no database or login service. The 11 MiB/4096² upload checks run before Modal, and Caddy caps the whole request at 12 MB.

## 1. Choose an immutable application ref

Use a reviewed application tag shaped `vN.N` or `vN.N.N`, or a full 40-character commit SHA. Branches, abbreviated SHAs, and model-asset tags are rejected.

```bash
export APP_REF='<approved-tag-or-40-character-sha>'
```

## 2. One-time host setup

Install Git, curl, Python 3.12, and a current Node.js LTS release that supports the pinned Next.js version. Keep ports 8000 and the retired 8501 closed in both the Oracle security list and the host firewall.

```bash
git clone https://github.com/Akshat-Tiwari69/Trace.git ~/Trace
cd ~/Trace
git fetch --tags origin
git checkout --detach "$APP_REF"

python3.12 -m venv .venv
./.venv/bin/pip install --upgrade pip==24.2
./.venv/bin/pip install -r deploy/requirements-app.txt

npm ci --prefix web --no-audit --no-fund
npm run build --prefix web
test -f web/out/index.html
```

Store the application ref and Modal credentials outside Git:

```bash
mkdir -p ~/.config/roadresilience ~/.config/systemd/user
printf 'DEPLOY_REF=%s\n' "$APP_REF" > ~/.config/roadresilience/deploy.env
cat > ~/.config/roadresilience/env <<'EOF'
MODAL_SEG_URL=<deployed-modal-endpoint>
MODAL_SEG_KEY=<shared-secret>
TRACE_ALLOWED_HOSTS=trace.tiwaribabu.in,localhost,127.0.0.1
EOF
chmod 600 ~/.config/roadresilience/deploy.env ~/.config/roadresilience/env

cp deploy/roadresilience.service ~/.config/systemd/user/
cp deploy/roadresilience-update.service ~/.config/systemd/user/
cp deploy/roadresilience-update.timer ~/.config/systemd/user/

export XDG_RUNTIME_DIR="/run/user/$(id -u)"
systemctl --user daemon-reload
systemctl --user enable --now roadresilience.service
systemctl --user enable --now roadresilience-update.timer
```

`roadresilience.service` runs one Uvicorn worker on `127.0.0.1:8000`. One worker is intentional: it owns the CPU queue and prevents concurrent graph analyses from starving the ARM host.

## 3. Caddy and logging

Install Caddy from its official Debian/Ubuntu repository, then install the checked-in configuration and journald cap:

```bash
sudo cp ~/Trace/deploy/Caddyfile /etc/caddy/Caddyfile
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy

sudo mkdir -p /etc/systemd/journald.conf.d
sudo cp ~/Trace/deploy/journald-roadresilience.conf /etc/systemd/journald.conf.d/
sudo systemctl restart systemd-journald
```

Only Caddy is public. `curl http://127.0.0.1:8000/healthz` should work on the host; an external connection to `:8000` must fail.

## 4. Modal GPU deployment

Use a separate operator environment; Torch is intentionally absent from the Oracle app environment.

```bash
cd ~/Trace
python3 -m venv .venv-modal
./.venv-modal/bin/pip install --upgrade pip==24.2
./.venv-modal/bin/pip install 'modal==1.5.1'
./.venv-modal/bin/modal setup
./.venv-modal/bin/modal deploy deploy/modal_app.py
```

Before deployment, verify:

- the `roadseg-key` Modal secret exists and is non-empty;
- `MODEL_SHA256` matches the intended `a4-roadseg-v3.2` release asset;
- the Modal code is from the same approved application ref;
- an invalid key and oversized payload fail before image decoding.

Rotate the shared key by updating Modal first, then `~/.config/roadresilience/env`, restarting the app, and running the upload smoke below.

## 5. Release, health gate, and rollback

Update `DEPLOY_REF` to the newly approved immutable ref, then trigger the systemd unit so it receives the environment file:

```bash
printf 'DEPLOY_REF=%s\n' "$APP_REF" > ~/.config/roadresilience/deploy.env
chmod 600 ~/.config/roadresilience/deploy.env
systemctl --user start roadresilience-update.service
journalctl --user -u roadresilience-update.service -n 100 --no-pager
```

`deploy/update.sh` resolves the ref to a commit, installs pinned Python packages, performs a clean npm install and static export, restarts the service, and polls `/healthz`. Any install, build, restart, or health failure restores the previous commit, dependencies, web export, and service.

Do not run `update.sh` directly unless the same `DEPLOY_REF` is exported in that shell.

## 6. Production verification

```bash
systemctl --user status roadresilience.service roadresilience-update.timer
journalctl --user -u roadresilience.service -n 100 --no-pager
curl -fsS http://127.0.0.1:8000/healthz
curl -fsS https://trace.tiwaribabu.in/healthz
curl -fsS https://trace.tiwaribabu.in/api/v1/aois/panaji_demo
```

Run one real, authorized upload (never sensitive imagery):

```bash
curl -fsS -D /tmp/trace-headers \
  -F 'image=@/path/to/public-test-crop.png;type=image/png' \
  -F 'resolution_m=0.5' \
  -F 'confirm_external_processing=true' \
  https://trace.tiwaribabu.in/api/v1/analyses
```

Poll the returned `status_url`, then open `result_url` and `graph_url`. Verify finite resilience in `[0,1]`, criticality rows, image-space GeoJSON, Modal cold/warm inference, queue recovery after an app restart, and browser rendering at mobile and desktop sizes.

The rollout is complete only when the checked-out commit equals the approved ref, the public page/API/upload succeed, 8000 and legacy 8501 are externally closed, and the deployed application ref, Modal ref, checkpoint SHA-256, and timestamp are recorded in the project tracker.
