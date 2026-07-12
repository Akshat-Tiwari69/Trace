# Deploying the Route Resilience dashboard (Oracle Always-Free ARM)

Public dashboard on the free Oracle box. It reads precomputed `data/sample/`
artifacts and runs CPU mask-to-graph/resilience analysis for uploaded masks, so
`requirements-app.txt` includes the minimal P2/P3 stack but not Torch or the
segmentation model. Heavy model inference lives elsewhere (see below).

Target box: Ubuntu 24.04 aarch64, user `ubuntu`. Runs as a **user** systemd service
(lingering already enabled) — no root needed except the one-time Caddy/journald setup.

## One-time bring-up (on the box)

```bash
# 0. venv module (Ubuntu ships python3 without venv)
sudo apt-get update && sudo apt-get install -y python3.12-venv

# 1. clone (public repo) + pick the deployed branch
git clone https://github.com/Akshat-Tiwari69/Trace.git ~/Trace
cd ~/Trace && git checkout v1.0.0       # use the approved immutable release tag

# 2. slim venv
python3 -m venv .venv
./.venv/bin/pip install -U pip
./.venv/bin/pip install -r deploy/requirements-app.txt

# 3. pin the immutable release ref and install the user services
mkdir -p ~/.config/roadresilience
printf 'DEPLOY_REF=v1.0.0\n' > ~/.config/roadresilience/deploy.env
mkdir -p ~/.config/systemd/user
cp deploy/roadresilience.service          ~/.config/systemd/user/
cp deploy/roadresilience-update.service   ~/.config/systemd/user/
cp deploy/roadresilience-update.timer     ~/.config/systemd/user/
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
systemctl --user daemon-reload
systemctl --user enable --now roadresilience.service
systemctl --user enable --now roadresilience-update.timer
```

## Network exposure — Caddy is the ONLY public entrypoint

Streamlit binds **127.0.0.1:8501** (loopback only — enforced in
`roadresilience.service`) and Caddy terminates TLS on **80/443** and proxies to it.

- **Do NOT open port 8501** — not in the box iptables, not in the Oracle
  Security List. Only **80 and 443** are opened (both places). If an old
  `8501` ingress rule or `iptables ACCEPT` exists from an earlier bring-up,
  **remove it** (Oracle console: VCN → subnet → Security List → delete the
  8501 ingress rule; box: `sudo iptables -L INPUT --line-numbers`, delete the
  8501 rule, `sudo netfilter-persistent save`).
- Caddy setup: `sudo apt-get install -y caddy`, copy `deploy/Caddyfile` to
  `/etc/caddy/Caddyfile`, `sudo systemctl restart caddy`. Cert is auto-provisioned
  for `trace.tiwaribabu.in`.
- The Caddyfile caps request bodies at **20MB**. **Rate limiting is NOT enabled**:
  it requires the non-standard [caddy-ratelimit](https://github.com/mholt/caddy-ratelimit)
  module (custom caddy build via `xcaddy`). Recommended follow-up.

## Journald size cap (do this once, needs root)

App + update logs go to the journal; cap it so it can never fill the boot volume:

```bash
sudo mkdir -p /etc/systemd/journald.conf.d
sudo cp ~/Trace/deploy/journald-roadresilience.conf /etc/systemd/journald.conf.d/
sudo systemctl restart systemd-journald
```

## Secrets on the box (`MODAL_SEG_URL` / `MODAL_SEG_KEY`)

The dashboard's upload→segment feature reads `MODAL_SEG_URL` and `MODAL_SEG_KEY`
from the environment. Keep them in an env file the service loads (never in the
repo), and make it owner-read-only:

```bash
chmod 600 <path-to-env-file>     # e.g. ~/.config/roadresilience/env
```

### Key-rotation runbook (`ROADSEG_KEY` / `MODAL_SEG_KEY`)

1. Generate a new random key (e.g. `openssl rand -hex 32`).
2. Update the Modal Secret: `modal secret create roadseg-key ROADSEG_KEY=<new>`
   (overwrites `roadseg-key`). New Modal containers pick it up; a `modal deploy`
   forces it immediately.
3. Update the box env file (`MODAL_SEG_KEY=<new>`), keep it `chmod 600`.
4. `systemctl --user restart roadresilience.service`.

Note: between steps 2 and 4 there is a **brief mismatch window** where uploads
fail with 401 — harmless for this app (retry the upload). For zero-downtime
rotation, extend the endpoint to accept **two keys** (current + next), roll the
box to the next key, then drop the old one.

## Auto-update (code pushed to GitHub → box self-updates)

`roadresilience-update.timer` runs `deploy/update.sh` every ~2 min: it `git fetch`es,
and **only if the deploy target moved** does it `git reset --hard` to it (the deploy
checkout is treated as read-only — no local commits), refresh deps, and restart.
It then polls `http://127.0.0.1:8501/_stcore/health` for ~30 s; **on failure it
rolls back** to the previous commit, reinstalls deps, restarts again, and logs
loudly to the journal.

- **`DEPLOY_REF` is required** and should name an immutable approved release tag
  (for example `v1.0.0`). Put it in
  `~/.config/roadresilience/deploy.env`; the updater refuses to deploy when it is
  absent, preventing accidental raw-`dev` production releases.
- Want *instant* deploys instead of ~2-min polling? Add a GitHub Actions job that
  SSHes in and runs `deploy/update.sh` on push (uses the already-open port 22).

## Model inference (fast path — off this box)

The ARM box is too slow for the SegFormer model (seconds/tile on 1 CPU core). Serve
inference from a **serverless GPU (Modal)** that scales to zero; the dashboard calls
it on demand. The Modal image bakes the `road_pan.pt` GitHub Release asset at build
time, **verified against the `MODEL_SHA256` pin in `deploy/modal_app.py`** — a new
release means: update `MODEL_SHA256` (from the local `models/road_pan.pt` hash),
then `modal deploy deploy/modal_app.py`.

### Tested versions (checkpoint compatibility)

The Modal image pins **torch 2.4.1**; training runs on **torch 2.12.1+cu126**.
The `road_pan.pt` checkpoint must stay loadable by **both** — don't adopt
torch-version-specific serialization features, and note the checkpoint format
requires `weights_only=False` at load time (it stores metadata alongside the
state dict). Re-verify a new checkpoint loads under the Modal pin before release.

## Python & dependency matrix

- App tested on **Python 3.11** (dev machines) and **3.12** (the box). A
  `.python-version` at repo root now pins **3.11** for dev/CI tooling
  (pyenv, `actions/setup-python`-style version detection) — this does **not**
  touch the box: `roadresilience.service` and `update.sh` both invoke
  `.venv/bin/...` directly, so neither one ever consults `.python-version`,
  and the box keeps running its own **Python 3.12**.
- geopandas is **aligned at 1.0.1** in both root `requirements.txt` (dev) and
  `deploy/requirements-app.txt` (prod) as of A42 (2026-07-10). The joint test
  that gated the flip: full suite (249 passed) under 1.0.1 in an isolated
  dev-side venv, plus a `gpd.read_file()` smoke on `data/sample/*.geojson` on
  the deploy box (which had already been serving on 1.0.1 in production).
  Apply the same joint-test rule to any future major bump of the geo stack.

## Useful ops

```bash
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
systemctl --user status roadresilience.service      # health
journalctl --user -u roadresilience.service -n 50   # app logs
journalctl --user -u roadresilience-update.service -n 50   # deploy/rollback logs
systemctl --user list-timers roadresilience-update.timer
~/Trace/deploy/update.sh                             # force an update now
curl -fsS http://127.0.0.1:8501/_stcore/health       # local health probe
```
