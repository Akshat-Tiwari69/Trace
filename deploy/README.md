# Deploying the Route Resilience dashboard (Oracle Always-Free ARM)

Public dashboard on the free Oracle box. **Dashboard only** — it reads precomputed
`data/sample/` artifacts and never loads the model, so it needs only the slim
`requirements-app.txt` (no torch/rasterio/osmnx). Heavy model inference lives
elsewhere (see "Model inference" below).

Target box: Ubuntu 24.04 aarch64, user `ubuntu`. Runs as a **user** systemd service
(lingering already enabled) — no root needed except a one-time firewall rule.

## One-time bring-up (on the box)

```bash
# 0. venv module (Ubuntu ships python3 without venv)
sudo apt-get update && sudo apt-get install -y python3.12-venv

# 1. clone (public repo) + pick the deployed branch
git clone https://github.com/Akshat-Tiwari69/Trace.git ~/Trace
cd ~/Trace && git checkout dev          # or the release branch you deploy from

# 2. slim venv
python3 -m venv .venv
./.venv/bin/pip install -U pip
./.venv/bin/pip install -r deploy/requirements-app.txt

# 3. install the user services (dashboard + auto-update timer)
mkdir -p ~/.config/systemd/user
cp deploy/roadresilience.service          ~/.config/systemd/user/
cp deploy/roadresilience-update.service   ~/.config/systemd/user/
cp deploy/roadresilience-update.timer     ~/.config/systemd/user/
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
systemctl --user daemon-reload
systemctl --user enable --now roadresilience.service
systemctl --user enable --now roadresilience-update.timer

# 4. open the port ON THE BOX (Oracle images REJECT by default in iptables)
sudo iptables -I INPUT 6 -p tcp --dport 8501 -j ACCEPT
sudo netfilter-persistent save      # persist across reboot (iptables-persistent)
```

### 5. Open the port in the Oracle console  (**manual — can't be scripted**)
VCN → your subnet → **Security List** → *Add Ingress Rule*:
Source `0.0.0.0/0`, IP Protocol TCP, **Destination port 8501**. Save.

Live at **http://<PUBLIC_IP>:8501**.

## Auto-update (code pushed to GitHub → box self-updates)

`roadresilience-update.timer` runs `deploy/update.sh` every ~2 min: it `git fetch`es
the tracked branch, and **only if origin moved** does it pull, refresh deps, and
`systemctl --user restart roadresilience`. No secrets, no inbound webhook port.

- Deployed branch = whatever is checked out in `~/Trace` (set upstream so `@{u}` resolves).
- Want *instant* deploys instead of ~2-min polling? Add a GitHub Actions job that
  SSHes in and runs `deploy/update.sh` on push (uses the already-open port 22).

## Model inference (fast path — off this box)

The ARM box is too slow for the SegFormer model (seconds/tile on 1 CPU core). Serve
inference from a **serverless GPU (Modal)** that scales to zero; the dashboard calls
it on demand. "Latest model" = the newest `road_pan.pt` GitHub Release asset; the
Modal function fetches that at build time, so a new release + `modal deploy` ships it.

## Useful ops

```bash
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
systemctl --user status roadresilience.service      # health
journalctl --user -u roadresilience.service -n 50   # app logs
systemctl --user list-timers roadresilience-update.timer
~/Trace/deploy/update.sh                             # force an update now
```
