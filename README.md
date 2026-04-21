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
  selection_error_column: metric_error_val
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
- `notebooks/03_hypothesis_check_nn_vs_gbdt.ipynb`
- `notebooks/04_methodology_walkthrough.ipynb`

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

## Configuration reference

### `version` (required)

Integer config/pipeline version. Also used in cache identity — bump it to force a full cache rebuild.

### `groups`

Define model families by their TabArena `config_type` values.

| Field | Type | Description |
|---|---|---|
| `config_types` | list of strings (required) | TabArena config type identifiers, e.g. `[CAT, GBM, XGB]` |
| `label` | string | Human-readable label for plots/tables. Defaults to the group key name |

### `comparisons`

One or more pairwise comparisons between groups. `delta_norm = group_a_error - group_b_error`, so positive values mean group_a performs worse.

| Field | Type | Values | Default |
|---|---|---|---|
| `name` | string (required) | Any identifier | — |
| `group_a` | string (required) | Must reference a key in `groups` | — |
| `group_b` | string (required) | Must reference a key in `groups` | — |
| `expected_direction` | string or null | `positive`, `negative`, `null` | `null` |

`expected_direction` controls one-sided vs two-sided testing:
- `null` — two-sided test and two-sided CI (use for exploratory analysis)
- `positive` — one-sided test, expects positive correlation; CI has lower bound only
- `negative` — one-sided test, expects negative correlation; CI has upper bound only

### `analysis`

| Field | Type | Values | Default |
|---|---|---|---|
| `unit` | string | `dataset`, `fold` | `dataset` |
| `error_column` | string | Any numeric column in TabArena results | `metric_error` |
| `selection_error_column` | string or null | Any numeric column, or `null` to reuse `error_column` | `metric_error_val` |
| `method_variant` | string | `default`, `tuned`, `tuned_ensemble` | `tuned` |
| `exclude_methods_containing` | list of strings | Substring patterns to exclude methods by name | `[]` |

- `unit: dataset` aggregates folds to dataset-level means before correlation (recommended for formal inference).
- `unit: fold` keeps every fold as a separate observation (inflates statistical power — folds share training data).
- `selection_error_column` controls which metric picks the best method per split (val-based selection). It defaults to `metric_error_val`.
- `error_column` is used for the final evaluation (test-based).
- Set `selection_error_column: null` only when you intentionally want selection and evaluation to use the same metric.

### `metafeatures`

| Field | Type | Values | Default |
|---|---|---|---|
| `feature_sets` | list of strings | `basic`, `irregularity`, `redundancy`, `pymfe` | `[basic, irregularity]` |
| `pymfe_groups` | list of strings | pymfe feature groups (e.g. `general`, `statistical`, `info-theory`) | `[general, statistical, info-theory]` |
| `pymfe_summary` | list of strings | pymfe summary functions (e.g. `mean`, `sd`, `min`, `max`) | `[mean, sd]` |
| `trace` | bool | `true`, `false` | `false` |
| `irregularity_components` | list of strings | See below | all five |

Available feature sets:
- `basic` — cheap, interpretable train-split descriptors:
  - size/composition: `n`, `d`, `log_n`, `log_d`, `n_over_d`, `d_over_n`, `n_num_features`,
    `n_cat_features`, `num_fraction`, `cat_fraction`
  - target structure for known `binary`/`multiclass` tasks: `n_classes`, `class_entropy`,
    `majority_class_fraction`, `minority_class_fraction`, `class_imbalance_ratio`
  - categorical cardinality: `mean_cat_cardinality`, `max_cat_cardinality`,
    `high_cardinality_fraction`, `cat_cardinality_to_n_ratio`
  - missingness: `missing_fraction`, `row_missing_fraction`, `feature_missing_fraction`,
    `num_missing_fraction`, `cat_missing_fraction`, `max_feature_missing_fraction`
  - numeric shape: `mean_abs_skew`, `max_abs_skew`, `mean_kurtosis`,
    `outlier_fraction_iqr`, `zero_fraction`
  - low-information columns: `constant_feature_fraction`, `near_constant_feature_fraction`
- `irregularity` — reproduction of the paper's 5-component composite on numeric columns. Components: `irreg_min_cov_eig` (min eigenvalue of the standardized covariance), `irreg_std_skew` (skewness of per-feature stds), `irreg_range_skew` (skewness of per-feature ranges), `irreg_iqr_hmean` (IQR of per-feature harmonic means, computed only over strictly-positive columns), `irreg_kurtosis_std` (std of per-feature kurtoses). The combined `irregularity` column is the weighted sum of per-component z-scores with the paper's coefficients `(-0.33, +0.23, +0.22, +0.21, +0.21)`. Z-scoring is performed across the datasets in the current analysis run.
- `redundancy` — opt-in numeric correlation/eigenvalue descriptors: `mean_abs_corr`, `max_abs_corr`,
  `high_corr_pair_fraction`, `effective_rank`, `participation_ratio`. It is skipped with `NaN`
  outputs above 512 non-constant numeric columns because it builds a full correlation matrix.
- `pymfe` — requires `pip install -e ".[pymfe]"`; extracts features via the pymfe library
- `trace: true` — keeps metafeature caches enabled and logs per-split feature-set timings, exact `pymfe` subgroup contents, per-output `pymfe` timings, and captured warning causes on cache misses. For readable ordering, use `parallelism.n_jobs: 1`.

Categorical handling note:
- Columns with dtype `object`, pandas `category`, or `bool` are treated as categorical.
- `basic` uses that classification for categorical counts, fractions, cardinality, and categorical missingness.
- `high_cardinality_fraction` counts categorical columns with at least `max(50, 0.1 * n_train)` observed non-missing values.
- `irregularity` runs on numeric columns only.
- `pymfe` internally converts categorical columns to category codes before extraction.

Missing-value handling for `pymfe`:
- `pymfe` does not receive raw missing values from the training split.
- Missing categorical values are filled with that column's mode, then the column is encoded as pandas category codes.
- Remaining columns are coerced to numeric, and missing numeric values are filled with that column's median.
- This preprocessing is only for `pymfe` extraction. The `basic` feature set still computes `missing_fraction` from the original, non-imputed training split.
- Some extracted `pymfe__*` values can still be `NaN` if `pymfe` cannot compute or summarize a feature after preprocessing.

Available irregularity components (all included by default):
- `irreg_min_cov_eig` — minimum covariance eigenvalue
- `irreg_std_skew` — standard deviation of feature skewness
- `irreg_range_skew` — range of feature skewness
- `irreg_kurtosis_std` — standard deviation of feature kurtosis

### `statistics`

| Field | Type | Values | Default |
|---|---|---|---|
| `correlation_method` | string | `spearman`, `pearson` | `spearman` |
| `alpha` | float | Significance threshold (0–1) | `0.05` |
| `fdr_method` | string or null | `bh`, `holm`, `null` | `bh` |
| `confidence_interval` | bool | `true`, `false` | `true` |
| `ci_bootstrap_samples` | int | Number of bootstrap resamples | `10000` |
| `ci_confidence_level` | float | CI coverage (0–1) | `0.95` |
| `multivariate` | bool | `true`, `false` | `false` |
| `multivariate_method` | string | `ols`, `ridge` | `ols` |

- `spearman` — rank-based, robust to outliers and non-linearity. Recommended for exploratory work.
- `pearson` — assumes linear relationship and normality.
- `bh` — Benjamini-Hochberg FDR correction (controls false discovery rate).
- `holm` — Holm-Bonferroni correction (controls family-wise error rate, more conservative).
- `multivariate` — only runs when exactly one comparison is configured. Fits all predictors jointly.

### `cache`

| Field | Type | Values | Default |
|---|---|---|---|
| `enabled` | bool | `true`, `false` | `true` |
| `directory` | string | Path relative to project root | `.mfa_cache` |
| `stages.raw_results` | bool | `true`, `false` | `true` |
| `stages.metafeatures` | bool | `true`, `false` | `true` |
| `stages.gaps` | bool | `true`, `false` | `true` |
| `stages.statistics` | bool | `true`, `false` | `true` |

Cache is content-hash based — config changes automatically produce fresh results. You only need to manually delete `.mfa_cache/` after **code changes** (the hash does not track source code).

---

## Methodology

For each configured comparison (`group_a` vs `group_b`), the package runs:

1. **Load and filter TabArena results**
   - Loads HPO result frames per method from `tabarena_context`.
   - Filters by `method_subtype == analysis.method_variant`.
   - Requires identifier columns `dataset`, `fold`, `method`, `config_type`, `method_subtype`, plus whichever metric columns are configured in `error_column` and `selection_error_column`.
   - **Imputed rows are treated as missing.** TabArena marks runs that failed (e.g. OOM, timeout) via an `imputed` flag and fills in a placeholder metric. The loader nulls those metric values while preserving the `imputed` / `impute_method` columns for auditing. Imputed rows never compete in best-in-group selection, and a split on which a family has only imputed candidates is dropped from that comparison. Rationale: placeholders are not observations; letting them win would bias the chosen representative, and silently using them understates the uncertainty on failure-prone splits.
   - **MNAR caveat.** Imputation is non-random — it correlates with things like dataset size (TabPFN OOMs on large `n`) or class imbalance. Dropping those splits biases the retained sample toward settings where the family actually runs, which can attenuate or bias meta-feature correlations. When a comparison loses many splits to imputation, sanity-check by intersecting on splits where **both** groups are real and comparing against the full retained set.

2. **Decode splits and normalize errors**
   - Decodes TabArena fold index into `(repeat, fold_in_repeat)`.
   - Computes selection-time normalized errors from `analysis.selection_error_column` (defaults to `metric_error_val`; falls back to `analysis.error_column` only when configured that way).
   - Computes evaluation-time normalized errors from `analysis.error_column`.

3. **Compute pairwise best-vs-best gaps per split**
   - For each split, picks the best method inside each group using selection metric (deterministic tie-break by selection normalized error, selection raw error, then method name).
   - Computes:
      - `delta_raw = best_a_error - best_b_error`
      - `delta_norm = best_a_norm_error - best_b_norm_error`
     where both deltas use the evaluation metric.
   - Positive deltas mean `group_a` has higher error than `group_b` for that split.

4. **Compute split-level meta-features**
   - `basic`: train-split size, dimensionality, target balance for known classification tasks,
     categorical cardinality, missingness, numeric shape, and constant/near-constant feature descriptors.
   - `irregularity`: component statistics on numeric columns, then a z-score-based irregularity proxy.
   - `redundancy` (optional): bounded numeric correlation/eigenvalue descriptors.
   - `pymfe` (optional): extracted through `pymfe` when enabled, after filling categorical missings with the mode, numeric missings with the median, and encoding categorical columns as integer codes.

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
