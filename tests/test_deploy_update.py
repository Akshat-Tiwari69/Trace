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
