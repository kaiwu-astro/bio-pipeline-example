import argparse

from numba import njit
import numpy as np


@njit(cache=True)
def _distances_matrix_kernel(encoded_sequences: np.ndarray) -> np.ndarray:
    n, sequence_length = encoded_sequences.shape
    distances = np.zeros((n, n), dtype=np.float64)
    gap = ord("-")

    for i in range(n):
        for j in range(n):
            dist = 0
            for k in range(sequence_length):
                c1 = encoded_sequences[i, k]
                c2 = encoded_sequences[j, k]
                if c1 != gap and c2 != gap and c1 != c2:
                    dist += 1
            distances[i, j] = dist
    return distances


def distances_matrix(sequences: list[str]) -> np.ndarray:
    """Calculate the matrix of distances between each pair of lines."""
    n = len(sequences)
    if n == 0:
        return np.zeros((0, 0))

    sequence_length = len(sequences[0])
    for sequence in sequences:
        assert len(sequence) == sequence_length

    encoded_sequences = np.zeros((n, sequence_length), dtype=np.uint8)
    for i, sequence in enumerate(sequences):
        encoded_sequences[i, :] = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)

    return _distances_matrix_kernel(encoded_sequences)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Path to aligned sequence dataset file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.dataset, "r") as f:
        sequences = [line.strip() for line in f if line.strip()]
    distances = distances_matrix(sequences)
    print(distances)


if __name__ == "__main__":
    main()
