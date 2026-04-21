"""pymfe feature classification by task type.

Source: pymfe documentation analysis.

Note: These lists assume pymfe is called with a discrete y (classification)
or continuous y (regression). Features marked regression-safe depend only on X.
"""

from __future__ import annotations

PYMFE_FILTER_SCHEMA_VERSION = 1

CLASSIFICATION_PROBLEM_TYPES: frozenset[str] = frozenset({"binary", "multiclass"})

PYMFE_CLASSIFICATION_ONLY: frozenset[str] = frozenset(
    {
        # general
        "freq_class",
        "nr_class",
        # statistical — MANOVA / class-aware
        "can_cor",
        "gravity",
        "lh_trace",
        "nr_disc",
        "p_trace",
        "roy_root",
        "sd_ratio",
        "w_lambda",
        # info-theory — discrete class required
        "class_conc",
        "class_ent",
        "eq_num_attr",
        "joint_ent",
        "mut_inf",
        "ns_ratio",
        # model-based — built on DecisionTreeClassifier
        "leaves",
        "leaves_branch",
        "leaves_corrob",
        "leaves_homo",
        "leaves_per_class",
        "nodes",
        "nodes_per_attr",
        "nodes_per_inst",
        "nodes_per_level",
        "nodes_repeated",
        "tree_depth",
        "tree_imbalance",
        "tree_shape",
        "var_importance",
        # landmarking — classifier performance metrics
        "best_node",
        "elite_nn",
        "linear_discr",
        "naive_bayes",
        "one_nn",
        "random_node",
        "worst_node",
        # clustering — class-label dependent indices
        "ch",
        "int",
        "nre",
        "pb",
        "sc",
        "sil",
        "vdb",
        "vdu",
        # concept — class-label variability
        "cohesiveness",
        "conceptvar",
        "impconceptvar",
        "wg_dist",
        # itemset — classification algorithm recommendation framework
        "one_itemset",
        "two_itemset",
        # complexity — Ho & Basu classification complexity framework
        "c1",
        "c2",
        "cls_coef",
        "density",
        "f1",
        "f1v",
        "f2",
        "f3",
        "f4",
        "hubs",
        "l1",
        "l2",
        "l3",
        "lsc",
        "n1",
        "n2",
        "n3",
        "n4",
        "t1",
    }
)

PYMFE_REGRESSION_SAFE: frozenset[str] = frozenset(
    {
        # general — X shape/type only
        "attr_to_inst",
        "cat_to_num",
        "inst_to_attr",
        "nr_attr",
        "nr_bin",
        "nr_cat",
        "nr_inst",
        "nr_num",
        "num_to_cat",
        # statistical — per-attribute descriptive stats on X
        "cor",
        "cov",
        "eigenvalues",
        "g_mean",
        "h_mean",
        "iq_range",
        "kurtosis",
        "mad",
        "max",
        "mean",
        "median",
        "min",
        "nr_cor_attr",
        "nr_norm",
        "nr_outliers",
        "range",
        "sd",
        "skewness",
        "sparsity",
        "t_mean",
        "var",
        # info-theory — attribute-only (no target involved)
        "attr_conc",
        "attr_ent",
        # complexity — PCA/dimensionality ratios on X
        "t2",
        "t3",
        "t4",
    }
)


def is_classification(problem_type: str | None) -> bool:
    if not isinstance(problem_type, str):
        return False
    return problem_type.lower() in CLASSIFICATION_PROBLEM_TYPES


def should_filter_classification_only(problem_type: str | None) -> bool:
    """Whether to drop classification-only pymfe features for this task.

    Only filters when `problem_type` is an explicit non-classification string
    (e.g. "regression"). `None` is treated as "unknown" and leaves the full
    feature set untouched — the pipeline always plumbs a known `problem_type`
    from tabarena metadata, so `None` only occurs in ad-hoc callers that
    intentionally bypass task-type awareness.
    """
    if not isinstance(problem_type, str):
        return False
    return problem_type.lower() not in CLASSIFICATION_PROBLEM_TYPES
