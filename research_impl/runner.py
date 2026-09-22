from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .adapter import load_reference
from .cache import build_k0
from .config import ROOT, git_provenance
from .evaluate import evaluate, retention_check, smoke_render
from .manifest import build_manifests
from .oracle import run_oracle_suite


RUN_ROOT = ROOT / "runs" / "research" / "P00"
MANIFEST_ROOT = ROOT / "research" / "manifests"
PROGRESS = ROOT / "research" / "progress.json"
LOCK = ROOT / "runs" / "research" / "gpu.lock"


@contextmanager
def gpu_lock():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    descriptor = None
    try:
        descriptor = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(descriptor, json.dumps({"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat()}).encode())
        os.close(descriptor)
        descriptor = None
        yield
    except FileExistsError as error:
        raise RuntimeError(f"GPU lock already exists: {LOCK}. Verify its PID before clearing it.") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if LOCK.exists():
            LOCK.unlink()


def initialize_progress() -> dict:
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text(encoding="utf-8"))
    payload = {
        "schema": "research-progress-v1",
        "authorized_stages": [0],
        "gpu_hours_consumed": 0.0,
        "cpu_selection_hours_consumed": 0.0,
        "resource_control_slots_used": 0,
        "queue": [{"task": "P00", "status": "RUNNING"}],
        "runs": {},
        "failures": [],
        "frozen_test_manifest": None,
        "git": git_provenance(),
    }
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Astra P00 shared research infrastructure")
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--hash-images", action="store_true")
    subparsers.add_parser("oracle")
    subparsers.add_parser("audit")
    subparsers.add_parser("smoke")
    subparsers.add_parser("retention-check")
    subparsers.add_parser("evaluate-reference")
    subparsers.add_parser("build-k0")
    args = parser.parse_args()
    initialize_progress()

    if args.command == "manifest":
        result = build_manifests(MANIFEST_ROOT, hash_images=args.hash_images)
    elif args.command == "oracle":
        result = run_oracle_suite(RUN_ROOT / "oracle.json")
    else:
        with gpu_lock():
            adapter, scene = load_reference(load_cameras=args.command != "audit" and args.command != "build-k0")
            if args.command == "audit":
                result = adapter.tensor_audit()
                RUN_ROOT.mkdir(parents=True, exist_ok=True)
                (RUN_ROOT / "interface_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            elif args.command == "smoke":
                result = smoke_render(adapter, scene, ["cam03"], [0, 149], RUN_ROOT / "smoke")
            elif args.command == "retention-check":
                result = retention_check(adapter, scene, "cam03", 149, RUN_ROOT / "retention_check.json")
            elif args.command == "evaluate-reference":
                samples = json.loads((MANIFEST_ROOT / "samples.json").read_text(encoding="utf-8"))
                splits = json.loads((MANIFEST_ROOT / "splits.json").read_text(encoding="utf-8"))
                result = evaluate(adapter, scene, splits["development"], samples["T24"], RUN_ROOT / "reference_v_t24")
            elif args.command == "build-k0":
                samples = json.loads((MANIFEST_ROOT / "samples.json").read_text(encoding="utf-8"))
                result = build_k0(adapter, samples["T24"], ROOT / "research_cache" / "cut_roasted_beef")
            else:
                raise AssertionError(args.command)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
