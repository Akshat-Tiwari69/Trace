"""Structural guard for the thin segmentation-training notebook."""

from __future__ import annotations

import ast
import json
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "train_segmentation.ipynb"


def test_training_notebook_delegates_without_reimplementing_trainer():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    tree = ast.parse(code)

    definitions = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    assert definitions == [], f"notebook must not duplicate trainer definitions: {definitions}"
    assert "src.pipeline.p1_segment.experiments.train_combined" in code
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "train_combined"
        for node in ast.walk(tree)
    )
    assert len(code.splitlines()) <= 60
