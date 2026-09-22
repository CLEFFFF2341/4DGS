from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "pretrained" / "cut_roasted_beef"
DEFAULT_SOURCE = ROOT / "data" / "cut_roasted_beef"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_namespace(path: Path) -> dict[str, Any]:
    """Parse an argparse ``Namespace(...)`` file without executing it."""
    tree = ast.parse(path.read_text(encoding="utf-8"), mode="eval")
    call = tree.body
    if not isinstance(call, ast.Call) or call.args:
        raise ValueError(f"Expected Namespace(keyword=literal, ...): {path}")
    if not isinstance(call.func, ast.Name) or call.func.id != "Namespace":
        raise ValueError(f"Only Namespace(...) is allowed: {path}")
    result: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            raise ValueError(f"Namespace ** expansion is not allowed: {path}")
        result[keyword.arg] = ast.literal_eval(keyword.value)
    return result


def reference_args(
    model_path: Path = DEFAULT_MODEL,
    source_path: Path = DEFAULT_SOURCE,
) -> SimpleNamespace:
    values = parse_namespace(model_path / "cfg_args")
    values["model_path"] = str(model_path.resolve())
    values["source_path"] = str(source_path.resolve())
    values["lazy_loader"] = True
    values["data_device"] = "cuda"
    values["eval"] = True
    return SimpleNamespace(**values)


def git_provenance() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
        ).strip()

    diff = run("diff", "--binary", "HEAD")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
        "dirty_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
    }
