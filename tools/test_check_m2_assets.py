#!/usr/bin/env python3
"""Focused fail-closed tests for check_m2_assets.py (metadata fixtures only)."""
from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import tempfile
import unittest

import check_m2_assets as checker


MLX = checker.PACKS["MLX 4-bit"]


class AssetCheckerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name) / "inputs"
        self.root.mkdir()
        self.old_root = checker.ROOT
        checker.ROOT = self.root
        self.old_cwd = os.getcwd()
        os.chdir(self.tmp.name)
        pathlib.Path("audit").mkdir()
        pathlib.Path("audit/M1-runlog.md").write_text("M1.8 verified\n")

    def tearDown(self) -> None:
        os.chdir(self.old_cwd)
        checker.ROOT = self.old_root
        self.tmp.cleanup()

    def write_pack(self, *, missing: bool = False, residual: int = 3, provenance: bool = True) -> pathlib.Path:
        directory = self.root / "mlx-q4"
        directory.mkdir()
        shard = "model-00001-of-00001.safetensors"
        index = directory / "model.safetensors.index.json"
        index.write_text(json.dumps({
            "metadata": {"total_size": MLX[3]},
            "weight_map": {"layer.0.weight": shard},
        }))
        if not missing:
            with (directory / shard).open("wb") as stream:
                stream.truncate(MLX[3] + residual)
        if provenance:
            lines = [f"# repo: {MLX[1]}", f"# revision: {MLX[2]}"]
            if not missing:
                import hashlib
                lines.append(f"{hashlib.sha256((directory / shard).read_bytes()).hexdigest()}  {shard}")
            import hashlib
            lines.append(f"{hashlib.sha256(index.read_bytes()).hexdigest()}  {index.name}")
            (directory / "SHA256SUMS").write_text("\n".join(lines) + "\n")
        return index

    def run_pack(self, index: pathlib.Path) -> tuple[bool, str]:
        output = io.StringIO()
        with contextlib.redirect_stderr(output), contextlib.redirect_stdout(output):
            result = checker.verify_pack("MLX 4-bit", index, MLX[3], MLX[1], MLX[2])
        return result, output.getvalue()

    def test_valid_index_missing_shard_names_shard(self) -> None:
        index = self.write_pack(missing=True)
        result, output = self.run_pack(index)
        self.assertFalse(result)
        self.assertIn("model-00001-of-00001.safetensors", output)

    def test_nonzero_residual_is_record_only(self) -> None:
        index = self.write_pack(residual=7)
        result, output = self.run_pack(index)
        self.assertTrue(result)
        self.assertIn("residual=7", output)
        self.assertIn("record-only", output)

    def test_unpinned_provenance_fails_closed(self) -> None:
        index = self.write_pack(provenance=True)
        manifest = index.parent / "SHA256SUMS"
        manifest.write_text(manifest.read_text().replace(MLX[2], "not-a-pinned-revision"))
        result, output = self.run_pack(index)
        self.assertFalse(result)
        self.assertIn("not pinned", output)


if __name__ == "__main__":
    unittest.main()
