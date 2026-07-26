"""Behavior checks for the immutable deployment-ref gate."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


def _run_update(tmp_path: Path, deploy_ref: str) -> subprocess.CompletedProcess[bytes]:
    home = tmp_path / "home"
    repo = home / "Trace"
    fake_bin = tmp_path / "bin"
    repo.mkdir(parents=True)
    fake_bin.mkdir()
    git = fake_bin / "git"
    git.write_text(
        "#!/usr/bin/env bash\n"
        "case \"$1\" in\n"
        "  fetch) exit 0 ;;\n"
        "  rev-parse)\n"
        "    [[ \"$DEPLOY_REF\" == v9.9 && \"$2\" != @ ]] && exit 1\n"
        "    printf '%s\\n' \"$FAKE_COMMIT\" ;;\n"
        "  *) exit 99 ;;\n"
        "esac\n",
        newline="\n",
    )
    git.chmod(0o755)

    env = os.environ.copy()
    env.update({
        "DEPLOY_REF": deploy_ref,
        "FAKE_COMMIT": "1" * 40,
        "HOME": home.as_posix(),
    })
    script = Path(__file__).parents[1] / "deploy" / "update.sh"
    bash = Path(os.environ["ProgramFiles"]) / "Git/bin/bash.exe" if os.name == "nt" else "bash"
    bin_path = (
        f"/{fake_bin.drive[0].lower()}{fake_bin.as_posix()[2:]}"
        if os.name == "nt" else fake_bin.as_posix()
    )
    return subprocess.run(
        [bash, "-c", 'export PATH="$1:$PATH"; bash "$2"', "_",
         bin_path, script.as_posix()],
        env=env, capture_output=True,
    )


@pytest.mark.parametrize("deploy_ref", [
    "dev", "main", "feature/next", "abc1234", "a4-roadseg-v3.2", "v1.2.3^{commit}",
])
def test_update_rejects_moving_or_unapproved_refs(tmp_path, deploy_ref):
    result = _run_update(tmp_path, deploy_ref)
    assert result.returncode == 2
    assert b"immutable application release tag or full 40-character commit SHA" in result.stderr


@pytest.mark.parametrize("deploy_ref", ["v1.0", "v1.2.3", "a" * 40])
def test_update_accepts_application_tags_and_full_shas(tmp_path, deploy_ref):
    assert _run_update(tmp_path, deploy_ref).returncode == 0


def test_update_rejects_release_tag_that_does_not_resolve(tmp_path):
    result = _run_update(tmp_path, "v9.9")
    assert result.returncode == 2
    assert b"does not resolve to a commit" in result.stderr


def test_update_builds_static_export_and_health_checks_fastapi():
    script = (Path(__file__).parents[1] / "deploy" / "update.sh").read_text()
    assert "npm ci --prefix web" in script
    assert "npm run build --prefix web" in script
    assert "test -f web/out/index.html" in script
    assert "http://127.0.0.1:8000/healthz" in script
    assert "http://127.0.0.1:8000/" in script
    assert "8501" not in script


def test_runtime_units_bind_fastapi_to_loopback_and_caddy():
    root = Path(__file__).parents[1]
    unit = (root / "deploy" / "roadresilience.service").read_text()
    caddy = (root / "deploy" / "Caddyfile").read_text()
    assert "uvicorn src.app.api:app" in unit
    assert "--host 127.0.0.1 --port 8000 --workers 1" in unit
    assert "EnvironmentFile=%h/.config/roadresilience/env" in unit
    assert "ReadWritePaths=%h/Trace/data/outputs" in unit
    assert "PrivateDevices=" not in unit
    update_unit = (root / "deploy" / "roadresilience-update.service").read_text()
    assert "ExecStart=/usr/bin/bash %h/Trace/deploy/update.sh" in update_unit
    assert "TimeoutStartSec=20min" in update_unit
    assert "reverse_proxy 127.0.0.1:8000" in caddy
    assert "Strict-Transport-Security" in caddy
    assert "8501" not in unit + caddy


def test_modal_endpoint_bounds_request_before_json_decode():
    source = (Path(__file__).parents[1] / "deploy" / "modal_app.py").read_text()
    assert "async for chunk in request.stream()" in source
    assert source.index("request.stream()") < source.index("json.loads(body)")
