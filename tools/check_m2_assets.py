#!/usr/bin/env python3
"""Fail-closed diagnostics for the pinned M2 packs and M1.8 evidence.

This checker only observes files.  It never downloads, creates, or substitutes assets.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path("inputs")
MLX_TOTAL = 16_054_262_240
BF16_TOTAL = 27_781_427_952
PACKS = {
    "MLX 4-bit": ("mlx-q4", "mlx-community/Qwen3.8-27B-4bit", "3e6447f082e89cc7f0bc6e5441afd38dfce760ff", MLX_TOTAL),
    "bf16": ("qwen-bf16", "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c4", BF16_TOTAL),
}


def blocker(message: str) -> None:
    print(f"M2 BLOCKED: {message}", file=sys.stderr)


def index_candidates(kind: str, root: pathlib.Path = ROOT) -> list[pathlib.Path]:
    candidates = list(root.rglob("model.safetensors.index.json")) if root.is_dir() else []
    def matches(path: pathlib.Path) -> bool:
        text = str(path).lower()
        if kind == "mlx":
            return any(t in text for t in ("mlx", "4bit", "4-bit", "q4"))
        return any(t in text for t in ("bf16", "bfloat16", "full", "qwen3.8-27b")) and not any(
            t in text for t in ("mlx", "4bit", "4-bit", "q4"))
    return [p for p in candidates if matches(p)]


def digest(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_provenance(directory: pathlib.Path, repo: str, revision: str, names: list[str], label: str) -> bool:
    manifest = directory / "SHA256SUMS"
    if not manifest.is_file():
        blocker(f"{label} required provenance is absent: {manifest}")
        return False
    headers: dict[str, str] = {}
    recorded: dict[str, str] = {}
    try:
        for line in manifest.read_text().splitlines():
            if line.startswith("# ") and ": " in line:
                key, value = line[2:].split(": ", 1)
                headers[key] = value
            elif len(line.split()) == 2:
                value, name = line.split()
                recorded[name] = value
    except OSError as exc:
        blocker(f"{label} provenance {manifest} is unreadable: {exc}")
        return False
    ok = True
    if headers.get("repo") != repo or headers.get("revision") != revision:
        blocker(f"{label} provenance is not pinned: expected repo={repo} revision={revision}; "
                f"observed repo={headers.get('repo', '<missing>')} revision={headers.get('revision', '<missing>')}")
        ok = False
    for name in names:
        path = directory / name
        if not path.is_file():
            continue
        if recorded.get(name) != digest(path):
            blocker(f"{label} provenance hash is missing or mismatched for {path}")
            ok = False
    return ok


def verify_pack(label: str, path: pathlib.Path, expected_total: int, repo: str, revision: str) -> bool:
    ok = True
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        blocker(f"{label} index {path} is unreadable or malformed: {exc}")
        return False
    try:
        weight_map = data["weight_map"]
        if not isinstance(weight_map, dict) or not weight_map:
            raise TypeError("weight_map must be a non-empty object")
        shard_names = sorted(set(weight_map.values()))
        if not all(isinstance(name, str) and name for name in shard_names):
            raise TypeError("weight_map values must be shard names")
    except (KeyError, TypeError) as exc:
        blocker(f"{label} index {path} has malformed weight_map: {exc}")
        return False
    try:
        declared = int(data["metadata"]["total_size"])
    except (KeyError, TypeError, ValueError) as exc:
        blocker(f"{label} index {path} has malformed metadata.total_size: {exc}")
        declared = None
        ok = False
    if declared is not None and declared != expected_total:
        blocker(f"{label} index {path} declares total_size={declared}, expected {expected_total}")
        ok = False
    missing = [name for name in shard_names if not (path.parent / name).is_file()]
    if missing:
        blocker(f"{label} pack is incomplete; missing shard(s) under {path.parent.resolve()}: {', '.join(missing)}")
        ok = False
    present = [name for name in shard_names if (path.parent / name).is_file()]
    observed = sum((path.parent / name).stat().st_size for name in present)
    residual = observed - declared if declared is not None else None
    print(f"M2 assets: {label} resolved index={path.resolve()} shard_sum={observed} "
          f"index_total={declared if declared is not None else '<invalid>'} residual={residual}")
    if declared is not None and not missing:
        # Residual is deliberately record-only (MLX has a pinned residual).
        print(f"M2 assets: {label} residual is record-only; shard bytes are not required to equal index total")
    ok = verify_provenance(path.parent, repo, revision, [path.name] + shard_names, label) and ok
    return ok


def main() -> int:
    ok = True
    root = ROOT.resolve()
    if not ROOT.is_dir():
        blocker(f"inputRoot {root} is missing; fetch both pinned M2 packs into inputs/ (never another path)")
    for label, (_dest, repo, revision, expected) in PACKS.items():
        kind = "mlx" if label.startswith("MLX") else "bf16"
        candidates = index_candidates(kind)
        if len(candidates) != 1:
            reason = "missing index" if not candidates else "expected exactly one index"
            blocker(f"{label} {reason} (model.safetensors.index.json); resolved candidate paths: "
                    f"{', '.join(str(p.resolve()) for p in candidates) if candidates else '<none under ' + str(root) + '>'}")
            ok = False
        else:
            ok = verify_pack(label, candidates[0], expected, repo, revision) and ok
    runlog = pathlib.Path("audit/M1-runlog.md")
    if not runlog.is_file():
        blocker(f"M1.8 evidence is missing: {runlog.resolve()}")
        ok = False
    else:
        try:
            text = runlog.read_text(errors="replace")
        except OSError as exc:
            blocker(f"M1.8 evidence {runlog.resolve()} is unreadable: {exc}")
            ok = False
        else:
            if "UNVERIFIED" in text or "M1.8 blocked" in text:
                blocker(f"M1.8 evidence remains unresolved in {runlog.resolve()}")
                ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
