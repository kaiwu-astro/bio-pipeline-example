# AGENTS.md

## Repository overview

This repository is a minimal bioinformatics example pipeline implemented in Python.

- `pipeline.py`: computes pairwise sequence distances and writes them to disk.
- `example/dataset.txt`: aligned sequence input consumed by `pipeline.py`.

## How to run

From the repository root:

```bash
python pipeline.py example/dataset.txt
```

## Linting, build, and tests

There are currently no configured lint or build commands in this repository.

When validating changes, run:

```bash
python pipeline.py example/dataset.txt
python -m unittest discover -s tests
```

and confirm both commands execute successfully.

## Change guidelines for agents

- Keep changes minimal and focused on the requested issue.
- Avoid introducing new tooling unless explicitly required.
- Preserve the current simple script-based workflow.
