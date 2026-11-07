import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from src.isp.data import safe_extract_zip


class ZipSafetyTests(unittest.TestCase):
    def test_safe_archive_extracts(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root/"work") as folder:
            payload=io.BytesIO()
            with zipfile.ZipFile(payload,"w") as z: z.writestr("train/raw/a.npy",b"sample")
            payload.seek(0); safe_extract_zip(payload,Path(folder)/"out")
            self.assertEqual((Path(folder)/"out"/"train"/"raw"/"a.npy").read_bytes(),b"sample")

    def test_path_traversal_is_rejected(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root/"work") as folder:
            payload=io.BytesIO()
            with zipfile.ZipFile(payload,"w") as z: z.writestr("../escape.csv",b"no")
            payload.seek(0)
            with self.assertRaises(ValueError): safe_extract_zip(payload,Path(folder)/"out")


if __name__=="__main__": unittest.main()
