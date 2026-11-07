import unittest
from pathlib import Path

from scripts.evaluate import resolve_test_root


class EvaluationPathTests(unittest.TestCase):
    def test_default_uses_bundled_synthetic_test_split(self):
        expected = Path(__file__).resolve().parents[1] / "data" / "synthetic_pairs" / "test"
        self.assertEqual(resolve_test_root(), expected.resolve())

    def test_dataset_root_resolves_to_test_subfolder(self):
        dataset_root = Path(__file__).resolve().parents[1] / "data" / "synthetic_pairs"
        self.assertEqual(resolve_test_root(str(dataset_root)), (dataset_root / "test").resolve())

    def test_missing_split_has_actionable_error(self):
        with self.assertRaisesRegex(SystemExit, "Expected matching RAW/RGB filenames"):
            resolve_test_root("data/not-a-dataset")


if __name__ == "__main__":
    unittest.main()
