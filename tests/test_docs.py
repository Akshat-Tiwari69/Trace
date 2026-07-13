"""Small, durable integrity checks for the current documentation set."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
CURRENT_DOCS = [
    ROOT / "AGENTS.md",
    ROOT / "CLAUDE.md",
    ROOT / "README.md",
    ROOT / "SETUP.md",
    ROOT / "bugs.md",
    ROOT / "deploy" / "README.md",
    *sorted((ROOT / "docs").glob("*.md")),
]
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^]]*]\(([^)]+)\)")


def test_local_markdown_links_resolve():
    missing: list[str] = []
    for document in CURRENT_DOCS:
        text = document.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or "://" in target or target.startswith(("mailto:", "#")):
                continue
            linked = (document.parent / unquote(target)).resolve()
            if not linked.exists():
                missing.append(f"{document.relative_to(ROOT)} -> {target}")
    assert missing == []


def test_agent_entrypoints_are_mirrors_after_their_headers():
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8").splitlines()
    claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8").splitlines()
    assert agents[4:] == claude[4:]
