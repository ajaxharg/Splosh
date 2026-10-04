#!/usr/bin/env python3
"""Fetch and verify Splosh input assets.

Large model packs are never fetched implicitly.  ``--fetch`` retains the M1
small-asset behaviour; use ``--fetch-pack mlx-q4``, ``--fetch-pack mlx-q8`` or
``--fetch-pack qwen-bf16`` explicitly when the corresponding disk space and
credentials are available. A pack fetched some other way (``hf download
--local-dir inputs/<pack>`` resumes and is faster) is adopted with
``--adopt-pack <pack>``, which checks every shard against the Hub's own
SHA-256 at the pinned revision before recording the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import sys
import urllib.error
import urllib.request

MLX_REPO = "mlx-community/Qwen3.8-27B-4bit"
MLX_REVISION = "3e6447f082e89cc7f0bc6e5441afd38dfce760ff"
MLX8_REPO = "mlx-community/Qwen3.8-27B-8bit"
MLX8_REVISION = "815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9"
BF16_REPO = "Qwen/Qwen3.8-27B"
BF16_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
# M1 compatibility names: the original default is the MLX repository.
REPO = MLX_REPO
REVISION = MLX_REVISION
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
CONFIG_FILE = "config.json"
PACKS = {
    "mlx-q4": {"repo": MLX_REPO, "revision": MLX_REVISION, "expected_total": 16_054_262_240, "residual": 279_109},
    "mlx-q8": {"repo": MLX8_REPO, "revision": MLX8_REVISION, "expected_total": 29_500_938_720, "residual": 279_759},
    "qwen-bf16": {"repo": BF16_REPO, "revision": BF16_REVISION, "expected_total": 27_781_427_952, "residual": None},
}


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def under_root(root: pathlib.Path, path: pathlib.Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def pack_directory(root: pathlib.Path, pack: str) -> pathlib.Path:
    if pack not in PACKS:
        raise ValueError(f"unknown pack {pack!r}")
    root = root.resolve()
    dest = (root / pack).resolve()
    if not under_root(root, dest) or dest == root:
        raise ValueError(f"pack destination outside inputRoot: {dest}")
    return dest


def download(repo: str, revision: str, name: str, dest: pathlib.Path) -> None:
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
    request = urllib.request.Request(url, headers={"User-Agent": "splosh-fetch-inputs/2"})
    token = os.environ.get("HF_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as out:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def write_manifest(directory: pathlib.Path, files: tuple[str, ...], provenance: dict[str, str]) -> None:
    lines = [f"# repo: {provenance['repo']}", f"# revision: {provenance['revision']}"]
    lines += [f"{sha256(directory / name)}  {name}" for name in files]
    (directory / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def verify(directory: pathlib.Path, files: tuple[str, ...], expected: dict[str, str] | None = None) -> int:
    sums = directory / "SHA256SUMS"
    if not sums.is_file():
        print(f"FAIL missing-hashes {sums}", file=sys.stderr); return 1
    recorded: dict[str, str] = {}
    for line in sums.read_text().splitlines():
        fields = line.split()
        if len(fields) == 2 and not line.startswith("#"): recorded[fields[1]] = fields[0]
    failed = False
    for name in files:
        path = directory / name
        if not path.is_file():
            print(f"FAIL missing-file {path}", file=sys.stderr); failed = True; continue
        actual = sha256(path)
        digest = recorded.get(name)
        if digest != actual or (expected and expected.get(name) not in (None, actual)):
            print(f"FAIL hash-mismatch {name}: expected {digest or '<missing>'}, observed {actual}", file=sys.stderr); failed = True
        else: print(f"OK {name} {actual}")
    return int(failed)


def verify_pack(directory: pathlib.Path, pack: str) -> int:
    """Verify revision metadata, exact shard set, and index totals.

    File byte sum and index ``total_size`` are deliberately reported as
    separate quantities.  MLX's known 279,109-byte residual is record-only.
    """
    if pack not in PACKS:
        print(f"FAIL unknown-pack {pack}", file=sys.stderr); return 1
    spec = PACKS[pack]
    directory = directory.resolve()
    index_path = directory / "model.safetensors.index.json"
    if not index_path.is_file():
        print(f"FAIL missing-index {index_path}", file=sys.stderr); return 1
    try:
        data = json.loads(index_path.read_text())
        values = data["weight_map"].values()
        expected_shards = sorted(set(values))
        declared = int(data["metadata"]["total_size"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"FAIL malformed-index {index_path}: {exc}", file=sys.stderr); return 1
    # The manifest is the local proof of the pinned revision and complete bytes.
    sums = directory / "SHA256SUMS"
    if not sums.is_file():
        print(f"FAIL missing-provenance {sums}", file=sys.stderr); return 1
    lines = sums.read_text().splitlines()
    headers = {line[2:].split(": ", 1)[0]: line[2:].split(": ", 1)[1]
               for line in lines if line.startswith("# ") and ": " in line}
    if headers.get("repo") != spec["repo"] or headers.get("revision") != spec["revision"]:
        print(f"FAIL unapproved-provenance {pack}: repo={headers.get('repo')} revision={headers.get('revision')}", file=sys.stderr)
        return 1
    recorded = {parts[1]: parts[0] for line in lines if not line.startswith("#")
                for parts in [line.split()] if len(parts) == 2}
    required_files = ["model.safetensors.index.json"] + expected_shards
    for name in required_files:
        path = directory / name
        if recorded.get(name) != sha256(path):
            print(f"FAIL hash-mismatch {pack}/{name}", file=sys.stderr); return 1
    actual_shards = sorted(p.name for p in directory.glob("*.safetensors"))
    if actual_shards != expected_shards:
        missing = sorted(set(expected_shards) - set(actual_shards))
        extra = sorted(set(actual_shards) - set(expected_shards))
        print(f"FAIL shard-set {pack}: missing={missing or []} extra={extra or []}", file=sys.stderr); return 1
    observed = sum((directory / name).stat().st_size for name in expected_shards)
    residual = observed - declared
    print(f"{pack} shard_sum={observed} index_total={declared} residual={residual}")
    if declared != spec["expected_total"]:
        print(f"FAIL index-total {pack}: expected {spec['expected_total']}, observed {declared}", file=sys.stderr); return 1
    if spec["residual"] is not None:
        print(f"{pack} expected residual (record-only)={spec['residual']}")
    return 0


def fetch_pack(root: pathlib.Path, pack: str) -> int:
    dest = pack_directory(root, pack)
    spec = PACKS[pack]
    print(f"required large-pack space: unknown (server shard sizes); available: {shutil.disk_usage(root).free} bytes")
    print(f"fetching explicitly requested {pack} into {dest}")
    try:
        download(spec["repo"], spec["revision"], "model.safetensors.index.json", dest / "model.safetensors.index.json")
        index = json.loads((dest / "model.safetensors.index.json").read_text())
        shards = sorted(set(index["weight_map"].values()))
        for name in shards:
            if pathlib.PurePosixPath(name).name != name or not name.endswith(".safetensors"):
                raise ValueError(f"unsafe shard name {name!r}")
            download(spec["repo"], spec["revision"], name, dest / name)
    except (OSError, ValueError, KeyError, urllib.error.URLError) as exc:
        print(f"FAIL download {exc}; assets are absent or incomplete", file=sys.stderr); return 1
    write_manifest(dest, tuple(["model.safetensors.index.json"] + shards), spec)
    return verify_pack(dest, pack)


def hub_hashes(repo: str, revision: str) -> dict[str, str]:
    """The Hub's SHA-256 of every LFS file of a repository at a revision."""
    request = urllib.request.Request(f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true",
                                     headers={"User-Agent": "splosh-fetch-inputs/2"})
    token = os.environ.get("HF_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=120) as response:
        listing = json.load(response)
    return {s["rfilename"]: s["lfs"]["sha256"] for s in listing.get("siblings", []) if s.get("lfs")}


def adopt_pack(root: pathlib.Path, pack: str) -> int:
    """Record the manifest of a pack already on disk, once its shards match the Hub's hashes."""
    dest = pack_directory(root, pack)
    spec = PACKS[pack]
    try:
        index = json.loads((dest / "model.safetensors.index.json").read_text())
        shards = sorted(set(index["weight_map"].values()))
        expected = hub_hashes(spec["repo"], spec["revision"])
    except (OSError, ValueError, KeyError, urllib.error.URLError) as exc:
        print(f"FAIL adopt {exc}", file=sys.stderr); return 1
    for name in shards:
        path = dest / name
        if not path.is_file():
            print(f"FAIL missing-file {path}", file=sys.stderr); return 1
        observed = sha256(path)
        if expected.get(name) != observed:
            print(f"FAIL hash-mismatch {pack}/{name}: hub {expected.get(name) or '<none>'}, observed {observed}", file=sys.stderr); return 1
        print(f"OK {name} {observed}")
    write_manifest(dest, tuple(["model.safetensors.index.json"] + shards), spec)
    return verify_pack(dest, pack)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fetch_inputs.py", description="Fetch and verify pinned Splosh model inputs.")
    p.add_argument("--check-access", action="store_true", help="check dependencies and writable input/artifact paths")
    p.add_argument("--fetch", action="store_true", help="download the pinned tokenizer and config assets")
    p.add_argument("--fetch-pack", choices=tuple(PACKS), help="explicitly download one large model pack (never implicit)")
    p.add_argument("--adopt-pack", choices=tuple(PACKS), help="record the manifest of a pack already on disk, after checking its shards against the Hub's hashes")
    p.add_argument("--verify", action="store_true", help="verify materialized files and recorded SHA256SUMS")
    p.add_argument("--pack", choices=tuple(PACKS), help="pack to verify (with --verify)")
    p.add_argument("--revision", help=argparse.SUPPRESS)
    p.add_argument("--repo", help=argparse.SUPPRESS)
    p.add_argument("--input-root", type=pathlib.Path, default=pathlib.Path("inputs"), help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    root = args.input_root
    if args.check_access:
        missing = [n for n in ("huggingface_hub", "tokenizers", "transformers") if importlib.util.find_spec(n) is None]
        for path in (root, pathlib.Path("artifacts")):
            path.mkdir(parents=True, exist_ok=True)
            if not os.access(path, os.W_OK): print(f"FAIL path-not-writable {path}", file=sys.stderr); return 1
        if missing: print("FAIL missing-python-packages " + ",".join(missing), file=sys.stderr); return 1
        print("PASS python-access"); return 0
    # Hidden compatibility switches are provenance guards, and must fail closed.
    # Left unset they are the selected pack's own, or the tokenizer's; set, they must agree.
    selected_pack = args.fetch_pack or args.adopt_pack or (args.pack if args.verify else None)
    if selected_pack:
        approved = PACKS[selected_pack]
        args.revision = args.revision or approved["revision"]
        args.repo = args.repo or approved["repo"]
        if args.revision != approved["revision"] or args.repo != approved["repo"]:
            print(f"FAIL unapproved-provenance repo={args.repo} revision={args.revision}", file=sys.stderr); return 1
    elif (args.revision or REVISION) != REVISION or (args.repo or REPO) != REPO:
        print(f"FAIL unapproved-provenance repo={args.repo} revision={args.revision}", file=sys.stderr); return 1
    args.revision = args.revision or REVISION
    args.repo = args.repo or REPO
    if args.fetch_pack:
        return fetch_pack(root, args.fetch_pack)
    if args.adopt_pack:
        try: return adopt_pack(root, args.adopt_pack)
        except ValueError as exc:
            print(f"FAIL {exc}", file=sys.stderr); return 1
    if args.pack and args.verify:
        try: return verify_pack(pack_directory(root, args.pack), args.pack)
        except ValueError as exc:
            print(f"FAIL {exc}", file=sys.stderr); return 1
    if args.fetch:
        try:
            for name in TOKENIZER_FILES + (CONFIG_FILE,):
                download(args.repo, args.revision, name, root / ("tokenizer" if name != CONFIG_FILE else "") / name)
        except (OSError, urllib.error.URLError) as exc:
            print(f"FAIL download {exc}", file=sys.stderr); return 1
        write_manifest(root / "tokenizer", TOKENIZER_FILES, {"repo": args.repo, "revision": args.revision})
        write_manifest(root, (CONFIG_FILE,), {"repo": args.repo, "revision": args.revision})
    if args.verify or args.fetch:
        if verify(root / "tokenizer", TOKENIZER_FILES): return 1
        return verify(root, (CONFIG_FILE,))
    p.print_help(); return 0


if __name__ == "__main__": raise SystemExit(main())
