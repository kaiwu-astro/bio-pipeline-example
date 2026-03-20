import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np

from pipeline import distances_matrix, write_distances_matrix_to_disk


class RegressionTest(unittest.TestCase):
    def test_pipeline_regression_output(self):
        repo_root = Path(__file__).resolve().parents[1]
        dataset_path = repo_root / "example" / "dataset.txt"
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "distances.npy"
            result = subprocess.run(
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

            expected_message = f"Wrote distance matrix to {output_path}\n"
            self.assertEqual(result.stdout, expected_message)

            matrix = np.load(output_path)
            expected = distances_matrix(
                [line.strip() for line in dataset_path.read_text().splitlines() if line.strip()]
            ).astype(np.uint16)
            np.testing.assert_array_equal(matrix, expected)

    def test_distances_matrix_gap_semantics(self):
        sequences = ["A-C", "ACC", "ATC"]
        result = distances_matrix(sequences)

        expected = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(result, result.T)

    def test_write_distances_matrix_to_disk_gap_semantics(self):
        sequences = ["A-C", "ACC", "ATC"]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "distances.npy"
            write_distances_matrix_to_disk(sequences, str(output_path))
            result = np.load(output_path)

        expected = np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
            ],
            dtype=np.uint16,
        )
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(result, result.T)


if __name__ == "__main__":
    unittest.main()
