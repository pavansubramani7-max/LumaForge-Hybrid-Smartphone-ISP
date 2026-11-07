import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from data.dataset import PairedRawRGBDataset, discover_pairs, guess_bayer_pattern, make_splits


class DatasetTests(unittest.TestCase):
    def setUp(self):
        workspace = Path(__file__).resolve().parents[1]
        self.temp = tempfile.TemporaryDirectory(dir=workspace / "work")
        self.root = Path(self.temp.name) / "images"
        for split in ("train", "val", "test"):
            (self.root / split / "raw").mkdir(parents=True)
            (self.root / split / "rgb").mkdir(parents=True)
        for index in range(10):
            split = "train" if index < 8 else "val" if index == 8 else "test"
            raw = np.zeros((32, 32), dtype=np.uint16)
            raw[0::2, 0::2] = 50000
            raw[1::2, 1::2] = 12000
            stem = f"scene_{index:03d}"
            np.save(self.root / split / "raw" / f"{stem}.npy", raw)
            target = np.full((32, 32, 3), index * 10, dtype=np.uint8)
            Image.fromarray(target).save(self.root / split / "rgb" / f"{stem}.png")

    def tearDown(self):
        self.temp.cleanup()

    def test_pairs_and_fixed_split_manifest(self):
        pairs = discover_pairs(self.root)
        self.assertEqual(len(pairs), 10)
        manifest = make_splits(self.root, self.root / "splits.json", seed=42)
        self.assertEqual(manifest["counts"]["total"], 10)
        self.assertEqual(sum(manifest["counts"][s] for s in ("train", "val", "test")), 10)

    def test_loader_returns_packed_raw_and_aligned_rgb(self):
        make_splits(self.root, self.root / "splits.json", seed=42)
        dataset = PairedRawRGBDataset(self.root, "train", self.root / "splits.json", patch_size=8, bayer_pattern="RGGB")
        sample = dataset[0]
        self.assertEqual(tuple(sample["raw"].shape), (4, 8, 8))
        self.assertEqual(tuple(sample["rgb"].shape), (3, 16, 16))

    def test_metadata_pattern_detection(self):
        result = guess_bayer_pattern(pattern=np.array([[0, 1], [1, 2]]))
        self.assertEqual(result["pattern"], "RGGB")
        self.assertEqual(result["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
