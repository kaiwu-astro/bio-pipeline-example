import subprocess
import sys
from pathlib import Path
import unittest

import numpy as np

from pipeline import distances_matrix


class RegressionTest(unittest.TestCase):
    def test_pipeline_regression_output(self):
        repo_root = Path(__file__).resolve().parents[1]
        dataset_path = repo_root / "example" / "dataset.txt"
        result = subprocess.run(
            [sys.executable, str(repo_root / "pipeline.py"), str(dataset_path)],
            capture_output=True,
            text=True,
            check=True,
        )

        expected_output = (
            "[[0. 2. 1. 0. 0. 0. 3. 1. 0. 2.]\n"
            " [2. 0. 1. 2. 2. 2. 1. 1. 2. 4.]\n"
            " [1. 1. 0. 1. 1. 1. 2. 0. 1. 3.]\n"
            " [0. 2. 1. 0. 0. 0. 3. 1. 0. 2.]\n"
            " [0. 2. 1. 0. 0. 0. 3. 1. 0. 2.]\n"
            " [0. 2. 1. 0. 0. 0. 3. 1. 0. 2.]\n"
            " [3. 1. 2. 3. 3. 3. 0. 2. 3. 5.]\n"
            " [1. 1. 0. 1. 1. 1. 2. 0. 1. 3.]\n"
            " [0. 2. 1. 0. 0. 0. 3. 1. 0. 2.]\n"
            " [2. 4. 3. 2. 2. 2. 5. 3. 2. 0.]]\n"
        )
        self.assertEqual(result.stdout, expected_output)

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


if __name__ == "__main__":
    unittest.main()
