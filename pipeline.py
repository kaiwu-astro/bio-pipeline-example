import argparse
import struct


def distance(seq1: str, seq2: str) -> int:
    """Calculate the distance between two strings.

    The distance is calculated by comparing each character in the two strings.
    If the characters are the same, or if either character is '-', then the distance is 0.
    Otherwise, the distance is 1.
    """
    assert len(seq1) == len(seq2)
    dist = 0
    for c1, c2 in zip(seq1, seq2):
        if c1 != '-' and c2 != '-' and c1 != c2:
            dist += 1
    return dist


def write_distances_upper_triangular(sequences: list[str], output: str) -> None:
    """Write upper-triangular pairwise distances to disk as uint16 values."""
    n = len(sequences)
    if n > 65535:
        raise ValueError("Number of sequences exceeds uint16 header capacity.")

    with open(output, "wb") as f:
        f.write(struct.pack("<H", n))
        for i, seq1 in enumerate(sequences):
            for seq2 in sequences[i:]:
                dist = distance(seq1, seq2)
                if dist > 65535:
                    raise ValueError("Distance exceeds uint16 range.")
                f.write(struct.pack("<H", dist))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Path to aligned sequence dataset file.")
    parser.add_argument(
        "--output",
        default="distances.bin",
        help="Path to output binary distance file (uint16 upper triangular matrix).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.dataset, "r") as f:
        sequences = [line.strip() for line in f if line.strip()]
    write_distances_upper_triangular(sequences, args.output)


if __name__ == "__main__":
    main()
