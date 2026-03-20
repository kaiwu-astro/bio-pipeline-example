import argparse

from numba import njit
import numpy as np

MAX_SEQUENCE_LENGTH = 65530


def _validate_sequence_lengths(sequences: list[str]) -> int:
    sequence_length = len(sequences[0])
    if sequence_length > MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"Sequence length {sequence_length} exceeds maximum supported length {MAX_SEQUENCE_LENGTH}."
        )
    for sequence in sequences:
        if len(sequence) != sequence_length:
            raise ValueError("All sequences must have the same length.")
    return sequence_length


@njit(cache=True)
def _distances_matrix_kernel(encoded_sequences: np.ndarray) -> np.ndarray:
    n, sequence_length = encoded_sequences.shape
    distances = np.zeros((n, n), dtype=np.float64)
    gap = ord("-")

    for i in range(n):
        for j in range(n):
            distances[i, j] = _pairwise_distance(encoded_sequences, i, j, sequence_length, gap)
    return distances


@njit(cache=True)
def _distance_row_kernel(encoded_sequences: np.ndarray, i: int) -> np.ndarray:
    n, sequence_length = encoded_sequences.shape
    distances = np.zeros(n, dtype=np.uint16)
    gap = ord("-")

    for j in range(n):
        distances[j] = _pairwise_distance(encoded_sequences, i, j, sequence_length, gap)
    return distances


@njit(cache=True)
def _pairwise_distance(encoded_sequences: np.ndarray, i: int, j: int, sequence_length: int, gap: int) -> int:
    dist = 0
    for k in range(sequence_length):
        c1 = encoded_sequences[i, k]
        c2 = encoded_sequences[j, k]
        if c1 != gap and c2 != gap and c1 != c2:
            dist += 1
    return dist


def distances_matrix(sequences: list[str]) -> np.ndarray:
    """Calculate the matrix of distances between each pair of lines."""
    n = len(sequences)
    if n == 0:
        return np.zeros((0, 0))

    sequence_length = _validate_sequence_lengths(sequences)

    encoded_sequences = np.zeros((n, sequence_length), dtype=np.uint8)
    for i, sequence in enumerate(sequences):
        encoded_sequences[i, :] = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)

    return _distances_matrix_kernel(encoded_sequences)


def write_distances_matrix_to_disk(sequences: list[str], output_path: str) -> None:
    """Calculate the matrix of distances and write it directly to disk as uint16."""
    n = len(sequences)
    if n == 0:
        matrix = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.uint16, shape=(0, 0))
        matrix.flush()
        return

    sequence_length = _validate_sequence_lengths(sequences)

    encoded_sequences = np.zeros((n, sequence_length), dtype=np.uint8)
    for i, sequence in enumerate(sequences):
        encoded_sequences[i, :] = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)

    matrix = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.uint16, shape=(n, n))
    for i in range(n):
        matrix[i, :] = _distance_row_kernel(encoded_sequences, i)
    matrix.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Path to aligned sequence dataset file.")
    parser.add_argument(
        "--output",
        default="distances.npy",
        help="Path to output .npy file for the distance matrix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        with open(args.dataset, "r") as f:
            sequences = [line.strip() for line in f if line.strip()]
        write_distances_matrix_to_disk(sequences, args.output)
        print(f"Wrote distance matrix to {args.output}")
    except ValueError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
