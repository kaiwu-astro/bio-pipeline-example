# AGENTS.md

## Repository overview

This repository is a minimal bioinformatics example pipeline implemented in Python.

- `pipeline.py`: computes pairwise sequence distances and prints a distance matrix.
- `dataset.txt`: aligned sequence input consumed by `pipeline.py`.

## How to run

From the repository root:

```bash
python pipeline.py
```

If `numpy` is not installed in your environment:

```bash
pip install numpy
```

## Linting, build, and tests

There are currently no configured lint, build, or automated test commands in this repository.

When validating changes, run:

```bash
python pipeline.py
```

and confirm it executes successfully and prints a matrix.

## Change guidelines for agents

- Keep changes minimal and focused on the requested issue.
- Avoid introducing new tooling unless explicitly required.
- Preserve the current simple script-based workflow.
