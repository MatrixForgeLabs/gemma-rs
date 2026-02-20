# Perf Reports

This folder stores baseline and comparison performance runs.

## Generate a baseline

```sh
scripts/perf/run_baseline.sh
```

Optional env vars:
- `GEMMA_PERF_ITERS_CORE` for `gemma-core` perf loops (default `20000`)
- `GEMMA_PERF_ITERS_OPS` for `gemma-ops` perf loops (default `500`)
- `GEMMA_PERF_COMPARE_TO` path to baseline markdown for regression checks
- `GEMMA_PERF_MAX_REGRESSION_PCT` allowed slowdown per metric (default `8`)
- `GEMMA_PERF_SUMMARY_FILE` TSV summary output path

The script writes a timestamped Markdown report:
- `reports/perf/baseline-<UTC timestamp>.md`
- `reports/perf/latest.md` (copy of last run)
- `reports/perf/latest-summary.tsv` (metric table)

## Regression gate example

```sh
GEMMA_PERF_COMPARE_TO=reports/perf/baseline-20260218T115321Z.md \
GEMMA_PERF_MAX_REGRESSION_PCT=8 \
scripts/perf/run_baseline.sh
```

Exit behavior:
- `0`: success, all metrics within threshold (or no compare baseline set)
- `2`: one or more metrics regressed beyond threshold
- `3`: compare baseline file not found
