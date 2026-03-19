import numpy as np

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


def distances_matrix(sequences: list[str]) -> np.ndarray:
    """Calculate the matrix of distances between each pair of lines."""
    n = len(sequences)
    distances = np.zeros((n,n))
    for i, seq1 in enumerate(sequences):
        for j, seq2 in enumerate(sequences):
            distances[i,j] = distance(seq1, seq2)
    return distances


def main():
    with open("dataset.txt", "r") as f:
        sequences = f.readlines()
    distances = distances_matrix(sequences)
    print(distances)


if __name__ == "__main__":
    main()
