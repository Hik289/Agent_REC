# Artifact Guide

Operational notes for reproducing `Learning to Recommend Multi-Agent Subgraphs from Calling Trees` from the public `Agent_REC` repository.

## Review Path

- `multi-agent system recommender/`: Project-specific implementation subtree.
- `single agent recommender/`: Project-specific implementation subtree.
- `data/`: Small fixtures, schemas, manifests, or data-layout notes; large data should stay outside git.
- `assets/`: README and paper-facing visual assets.

## Environment Files

- `requirements.txt`: Primary Python dependency list.

## Smoke Checks

Run these checks before long jobs:

```bash
python -m compileall -q .
```

If no smoke command is tracked, use the README Quick Start with the smallest seed, sample, or task count.

## Reproduction Entry Points

No single reproduction runner is tracked. Use the README commands and keep first runs small before full grids.

## Data Layout Notes

- `data/README.md`

## Figure Assets

- `assets/pipeline.png`
- `assets/t.png`

## Data And Outputs

- Keep local dataset paths, downloaded corpora, checkpoints, and generated run artifacts outside git unless the README identifies them as small checked-in fixtures.
- Record dataset version, preprocessing command, seed, and hardware/runtime notes for every reproduced table or figure.
- Treat generated JSONL files, logs, caches, model checkpoints, and benchmark downloads as local artifacts unless explicitly tracked as fixtures.
- For stochastic experiments, record seeds, task counts, dataset splits, and the exact git commit used for the run.

## Reporting Checklist

- `git rev-parse HEAD`
- Python version and dependency-install command
- Full command line for every table, figure, or benchmark cell
- Paths to raw outputs and aggregation scripts
- External data, benchmark, or API-backed steps that were intentionally skipped
