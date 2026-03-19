import subprocess
import struct
import sys
import tempfile
from pathlib import Path
import unittest


class RegressionTest(unittest.TestCase):
    def test_pipeline_writes_expected_binary_output(self):
        repo_root = Path(__file__).resolve().parents[1]
        dataset_path = repo_root / "example" / "dataset.txt"
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "distances.bin"
            subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "pipeline.py"),
                    str(dataset_path),
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            self.assertTrue(output_path.exists())
            raw = output_path.read_bytes()
            n = struct.unpack("<H", raw[:2])[0]
            self.assertEqual(n, 10)
            upper_count = n * (n + 1) // 2
            expected_size = 2 + upper_count * 2
            self.assertEqual(len(raw), expected_size)

            upper = struct.unpack(f"<{upper_count}H", raw[2:])
            matrix = [[0 for _ in range(n)] for _ in range(n)]
            idx = 0
            for i in range(n):
                matrix[i][i] = upper[idx]
                idx += 1
                for j in range(i + 1, n):
                    matrix[i][j] = upper[idx]
                    matrix[j][i] = upper[idx]
                    idx += 1

            expected_matrix = [
                [0, 2, 1, 0, 0, 0, 3, 1, 0, 2],
                [2, 0, 1, 2, 2, 2, 1, 1, 2, 4],
                [1, 1, 0, 1, 1, 1, 2, 0, 1, 3],
                [0, 2, 1, 0, 0, 0, 3, 1, 0, 2],
                [0, 2, 1, 0, 0, 0, 3, 1, 0, 2],
                [0, 2, 1, 0, 0, 0, 3, 1, 0, 2],
                [3, 1, 2, 3, 3, 3, 0, 2, 3, 5],
                [1, 1, 0, 1, 1, 1, 2, 0, 1, 3],
                [0, 2, 1, 0, 0, 0, 3, 1, 0, 2],
                [2, 4, 3, 2, 2, 2, 5, 3, 2, 0],
            ]
            self.assertEqual(matrix, expected_matrix)


if __name__ == "__main__":
    unittest.main()
