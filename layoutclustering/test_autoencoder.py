#!/usr/bin/env python3
"""Tests for AutoEncoder manifest and feature-vector export helpers."""

from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import layout_clustering_autoencoder as ae


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, image=np.asarray(image, dtype=np.float32))


class AutoEncoderTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = SCRIPT_DIR / "test_outputs" / "_autoencoder_unit"
        shutil.rmtree(self.temp_root, ignore_errors=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def _manifest(self, count=3, *, include_layout=False, mismatch=False):
        manifest = self.temp_root / "manifest.jsonl"
        rows = []
        for idx in range(count):
            image = np.full((8, 8), float(idx) / 10.0, dtype=np.float32)
            if mismatch and idx == count - 1:
                image = np.zeros((10, 8), dtype=np.float32)
            aerial = self.temp_root / f"s{idx}_aerial.npz"
            _write_image(aerial, image)
            row = {
                "sample_id": f"s{idx}",
                "source_path": "unit.oas",
                "marker_id": f"m{idx}",
                "clip_bbox": [0.0, 0.0, 1.0, 1.0],
                "aerial_npz": str(aerial),
            }
            if include_layout:
                layout = self.temp_root / f"s{idx}_layout.npz"
                _write_image(layout, image)
                row["layout_npz"] = str(layout)
            rows.append(row)
        with manifest.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return manifest

    def test_manifest_npz_loads_aerial_only(self):
        dataset = ae._ImageManifestDataset(self._manifest())

        self.assertEqual(dataset.channels, ["aerial"])
        self.assertEqual(dataset.shape, (1, 8, 8))
        stack, sample_id = dataset[0]
        self.assertEqual(sample_id, "s0")
        self.assertEqual(stack.shape, (1, 8, 8))

    def test_multichannel_shape_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "same channel/height/width"):
            ae._ImageManifestDataset(self._manifest(mismatch=True))

    def test_parser_has_train_and_encode_subcommands(self):
        parser = ae._build_parser()
        train_args = parser.parse_args(
            [
                "train",
                "--manifest",
                "train.jsonl",
                "--model-out",
                "ae.pt",
            ]
        )
        encode_args = parser.parse_args(
            [
                "encode",
                "--manifest",
                "all.jsonl",
                "--model",
                "ae.pt",
                "--features-out",
                "features.npz",
                "--fv-manifest-out",
                "fv.jsonl",
            ]
        )

        self.assertEqual(train_args.latent_dim, 128)
        self.assertEqual(encode_args.batch_size, 128)

    def test_train_and_encode_smoke(self):
        manifest = self._manifest(count=2)
        model = self.temp_root / "ae.pt"
        features = self.temp_root / "features.npz"
        fv_manifest = self.temp_root / "fv.jsonl"
        parser = ae._build_parser()

        train_args = parser.parse_args(
            [
                "train",
                "--manifest",
                str(manifest),
                "--model-out",
                str(model),
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--latent-dim",
                "4",
                "--device",
                "cpu",
            ]
        )
        encode_args = parser.parse_args(
            [
                "encode",
                "--manifest",
                str(manifest),
                "--model",
                str(model),
                "--features-out",
                str(features),
                "--fv-manifest-out",
                str(fv_manifest),
                "--batch-size",
                "2",
                "--device",
                "cpu",
            ]
        )

        self.assertEqual(ae.train(train_args), 0)
        self.assertEqual(ae.encode(encode_args), 0)
        with np.load(features, allow_pickle=False) as data:
            self.assertEqual(data["features"].shape, (2, 4))
            self.assertEqual(data["sample_ids"].astype(str).tolist(), ["s0", "s1"])


if __name__ == "__main__":
    unittest.main()
