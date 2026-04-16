# Meta-Feature Analysis

`meta-feature-analysis` is a Python package for running pairwise TabArena meta-feature analyses:

1. Build pairwise performance gaps between model groups (for example, **NN vs GBDT**).
2. Join those gaps with dataset/split meta-features.
3. Run statistical tests to identify which meta-features are associated with relative performance.

---

## Installation

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[dev]"      # pytest + ruff
pip install -e ".[pymfe]"    # optional pymfe feature set
```

If caching is enabled (default), make sure a parquet engine is available (for example `pyarrow`).

---

## Quick tutorial (end-to-end)

### 1) Start from a config

Use the provided template at `configs/default.yaml` or create your own.

Minimal structure:

```yaml
version: 1

groups:
  gbdt:
    config_types: [CAT, GBM, XGB, PB]
    label: "GBDT"
  nn:
    config_types: [FASTAI, NN_TORCH]
    label: "NN"

comparisons:
  - name: nn_vs_gbdt
    group_a: nn
    group_b: gbdt
    expected_direction: positive

analysis:
  unit: dataset
  error_column: metric_error
  method_variant: tuned

metafeatures:
  feature_sets: [basic, irregularity]

statistics:
  correlation_method: spearman
  alpha: 0.05
  fdr_method: bh

cache:
  enabled: true
  directory: .mfa_cache
```

### 2) Run analysis

```python
from pathlib import Path
from mfa import load_config, run_analysis

config = load_config(Path("configs/default.yaml"))
result = run_analysis(config, datasets=["diabetes", "credit-g"])
```

### 3) Inspect outputs

```python
result.analysis_table.head()      # merged table used for statistics
result.gap_table.head()           # pairwise gaps per split
result.metafeature_table.head()   # computed meta-features
```

Convert correlation results to a DataFrame:

```python
import pandas as pd

corr_df = pd.DataFrame([r.__dict__ for r in result.correlation_results])
if result.correction_result is not None:
    corr_df["p_value_adj"] = result.correction_result.adjusted_p_values
    corr_df["rejected"] = result.correction_result.rejected

corr_df.sort_values("p_value").head(20)
```

### 4) Optional: custom comparison in code

```python
import yaml
from mfa.config import parse_config
from mfa.pipeline import run_analysis

raw_config = yaml.safe_load(open("configs/default.yaml", "r", encoding="utf-8"))
raw_config["comparisons"] = [
    {
        "name": "tree_vs_nn",
        "group_a": "tree",
        "group_b": "nn",
        "expected_direction": "positive",
    }
]

config = parse_config(raw_config)
result = run_analysis(config, datasets=["diabetes", "credit-g"])
```

Reference notebooks:
- `notebooks/01_gbdt_vs_nn.ipynb`
- `notebooks/02_custom_comparison.ipynb`

---

## API summary

Public imports:

```python
from mfa import load_config, run_analysis
```

- `load_config(path) -> AnalysisConfig`: validates YAML and returns a typed config object.
- `run_analysis(config, datasets=None, task_metadata=None, tabarena_context=None) -> AnalysisResult`

`AnalysisResult` includes:
- `gap_table`
- `metafeature_table`
- `analysis_table`
- `correlation_results`
- `correction_result`
- `multivariate_result`

---

## Configuration guide

Top-level config sections:

- `version`: integer config/pipeline version (also used in cache identity).
- `groups`: model families defined by `config_types`.
- `comparisons`: one or more pairwise comparisons (`group_a` vs `group_b`).
- `analysis`: split unit, method subtype filter, and result column names.
- `metafeatures`: selected feature sets (`basic`, `irregularity`, optional `pymfe`).
- `statistics`: correlation, confidence intervals, FDR correction, optional multivariate model.
- `cache`: cache toggle, directory, and per-stage cache control.

Important analysis options:

- `analysis.unit`:
  - `dataset` (default): aggregate over splits before correlation.
  - `fold`: keep split-level rows (warning: folds are not independent observations).
- `analysis.method_variant`: one of `default`, `tuned`, `tuned_ensemble`.
- `statistics.fdr_method`: `bh`, `holm`, or `null` (disable correction).
- `statistics.multivariate`: only runs when exactly one comparison is configured.

---

## Methodology

For each configured comparison (`group_a` vs `group_b`), the package runs:

1. **Load and filter TabArena results**
   - Loads HPO result frames per method from `tabarena_context`.
   - Filters by `method_subtype == analysis.method_variant`.
   - Requires columns: `dataset`, `fold`, `method`, `metric_error`, `metric_error_val`, `config_type`, `method_subtype`.

2. **Decode splits and normalize errors**
   - Decodes TabArena fold index into `(repeat, fold_in_repeat)`.
   - Computes `norm_error` with TabArena `NormalizedScorer` per `(dataset, split_id)`.

3. **Compute pairwise best-vs-best gaps per split**
   - For each split, picks the best method inside each group (deterministic tie-break by normalized error, raw error, then method name).
   - Computes:
     - `delta_raw = best_a_error - best_b_error`
     - `delta_norm = best_a_norm_error - best_b_norm_error`
   - Positive deltas mean `group_a` has higher error than `group_b` for that split.

4. **Compute split-level meta-features**
   - `basic`: `n`, `d`, `log_n`, `n_over_d`, `cat_fraction`, `missing_fraction`.
   - `irregularity`: component statistics on numeric columns, then a z-score-based irregularity proxy.
   - `pymfe` (optional): extracted through `pymfe` when enabled.

5. **Join and aggregate analysis table**
   - Merges meta-features with gap rows on `(dataset, repeat, fold)`.
   - If `analysis.unit=dataset`, aggregates by dataset/comparison (including `n_splits`, mean deltas, std, sem).

6. **Univariate correlation testing**
   - Runs Spearman or Pearson correlations for each predictor vs `delta_norm` inside each comparison.
   - Supports one-sided p-values when `expected_direction` is configured.
   - Optional bootstrap confidence intervals for correlation coefficients.

7. **Multiple testing correction**
   - Applies BH or Holm correction **per comparison** over predictor p-values.

8. **Optional multivariate model**
   - If enabled and exactly one comparison is present, runs OLS or ridge with all selected predictors.
   - Returns coefficients, p-values (OLS), R² metrics, and VIF.

---

## Reproducibility

To record which TabArena revision was used:

```bash
git -C ../tabarena rev-parse HEAD
```

Also store:
- the full analysis YAML,
- the package version,
- and the generated `config_hash` from `AnalysisResult`.
