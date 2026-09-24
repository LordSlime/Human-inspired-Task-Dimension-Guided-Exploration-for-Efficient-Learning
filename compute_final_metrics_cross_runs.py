import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_CONFIG = "cross_run_metric_config.json"
DEFAULT_SAVE_DIR = "./paper_figures"
DEFAULT_NUM_RUNS = 5


ALG_TRIPLETS = [
    ("Linear", "LinUCB", "LowD_HieRec_LinearUCB_NoFD", "LowD_HieRec_LinearUCB"),
    ("NeuralUCB", "NeuralUCB", "LowD_HieRec_NeuralUCB_NoFD", "LowD_HieRec_NeuralUCB"),
    ("NeuralTS", "NeuralTS", "LowD_HieRec_NeuralTS_NoFD", "LowD_HieRec_NeuralTS"),
]

IC_PAIRS = [
    ("Linear", "ItemCluster_LinearUCB", "LowD_HieRec_LinearUCB"),
    ("NeuralUCB", "ItemCluster_NeuralUCB", "LowD_HieRec_NeuralUCB"),
    ("NeuralTS", "ItemCluster_NeuralTS", "LowD_HieRec_NeuralTS"),
]

FILTER_PAIRS = [
    ("Linear", "LowD_HieRec_LinearUCB", "LowD_HieRec_LinearUCB_Fusion"),
    ("NeuralUCB", "LowD_HieRec_NeuralUCB", "LowD_HieRec_NeuralUCB_Fusion"),
    ("NeuralTS", "LowD_HieRec_NeuralTS", "LowD_HieRec_NeuralTS_Fusion"),
]

H2N_PAIRS = [
    ("NeuralUCB", "H2N_Bandit", "LowD_HieRec_NeuralUCB"),
]

COFINE_PAIRS = [
    ("Linear", "CoFineUCB", "LowD_HieRec_LinearUCB"),
]

DEPTH_PAIRS = [
    ("Linear", "LowD_HieRec_LinearUCB", "LowD_HieRec_LinearUCB_4L"),
    ("NeuralUCB", "LowD_HieRec_NeuralUCB", "LowD_HieRec_NeuralUCB_4L"),
    ("NeuralTS", "LowD_HieRec_NeuralTS", "LowD_HieRec_NeuralTS_4L"),
]

METRICS = [
    {
        "name": "Online Training Cumulative Regret",
        "mean_pattern": "mean_regret_{alg}.npy",
        "std_pattern": "std_regret_{alg}.npy",
        "lower_is_better": True,
    },
    {
        "name": "Cold-Start Cumulative Regret",
        "mean_pattern": "mean_cs_cum_regret_{alg}.npy",
        "std_pattern": "std_cs_cum_regret_{alg}.npy",
        "lower_is_better": True,
    },
    {
        "name": "Cold-Start Final-Round Regret",
        "mean_pattern": "mean_cs_avg_regret_{alg}.npy",
        "std_pattern": "std_cs_avg_regret_{alg}.npy",
        "lower_is_better": True,
    },
    {
        "name": "Cold-Start Average Reward",
        "mean_pattern": "mean_cs_reward_{alg}.npy",
        "std_pattern": "std_cs_reward_{alg}.npy",
        "lower_is_better": False,
    },
]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_results_dir(config, dataset, alg):
    dataset_cfg = config["datasets"].get(dataset, {})
    alg_dirs = dataset_cfg.get("alg_dirs", {})
    if alg in alg_dirs:
        return alg_dirs[alg]
    return dataset_cfg.get("default_dir")


def load_final(results_dir, alg_key, metric, num_runs,
               saved_std_ddof=0):
    """Load the final mean and the standard deviation stored on disk."""
    if not results_dir:
        return None, None, None

    mean_path = Path(results_dir) / metric["mean_pattern"].format(alg=alg_key)
    std_path = Path(results_dir) / metric["std_pattern"].format(alg=alg_key)
    if not mean_path.exists():
        return None, None, str(mean_path)

    mean_arr = np.load(mean_path)
    if len(mean_arr) == 0:
        return None, None, str(mean_path)

    final_mean = float(mean_arr[-1])
    final_std = None
    if std_path.exists():
        std_arr = np.load(std_path)
        if len(std_arr) > 0:
            final_std = float(std_arr[-1])
    return final_mean, final_std, str(mean_path)


def improvement(base_val, new_val, lower_is_better):
    if base_val is None or new_val is None or base_val == 0:
        return None
    if lower_is_better:
        return (base_val - new_val) / abs(base_val) * 100
    return (new_val - base_val) / abs(base_val) * 100


def pvalue_from_summary(left_mean, left_std, right_mean, right_std,
                        lower_is_better, num_runs, saved_std_ddof):
    """Two-sided Welch test, converting population SD inputs when needed."""
    if (left_mean is None or right_mean is None or left_std is None
            or right_std is None or num_runs is None or num_runs < 2):
        return None

    scale = np.sqrt(num_runs / (num_runs - 1)) if saved_std_ddof == 0 else 1.0
    result = stats.ttest_ind_from_stats(
        mean1=float(left_mean),
        std1=float(left_std) * scale,
        nobs1=int(num_runs),
        mean2=float(right_mean),
        std2=float(right_std) * scale,
        nobs2=int(num_runs),
        equal_var=False,
        alternative="two-sided",
    )
    if np.isnan(result.pvalue):
        return None
    return float(result.pvalue)


def fmt_pvalue(p_value):
    if p_value is None:
        return "N/A"
    if p_value < 1e-4:
        return "<1e-4"
    return f"{p_value:.4f}"


def sig_stars(p_value):
    if p_value is None:
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def fmt_val(mean, std):
    if mean is None:
        return "N/A"
    if std is None:
        return f"{mean:.3f}"
    return f"{mean:.3f}+/-{std:.3f}"


def fmt_pct(pct):
    if pct is None:
        return "N/A"
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def append_pair_row(rows, section, dataset, family, metric, left_label,
                    left_alg, right_label, right_alg, config, num_runs,
                    saved_std_ddof):
    left_dir = resolve_results_dir(config, dataset, left_alg)
    right_dir = resolve_results_dir(config, dataset, right_alg)
    left_mean, left_std, left_path = load_final(
        left_dir, left_alg, metric, num_runs, saved_std_ddof)
    right_mean, right_std, right_path = load_final(
        right_dir, right_alg, metric, num_runs, saved_std_ddof)
    pct = improvement(left_mean, right_mean, metric["lower_is_better"])
    p_value = pvalue_from_summary(
        left_mean, left_std, right_mean, right_std,
        metric["lower_is_better"], num_runs, saved_std_ddof)

    rows.append({
        "section": section,
        "dataset": dataset,
        "family": family,
        "metric": metric["name"],
        "left_label": left_label,
        "left_alg": left_alg,
        "left_mean": left_mean,
        "left_std": left_std,
        "left_dir": left_dir,
        "left_path": left_path,
        "right_label": right_label,
        "right_alg": right_alg,
        "right_mean": right_mean,
        "right_std": right_std,
        "right_dir": right_dir,
        "right_path": right_path,
        "delta_left_to_right_pct": pct,
        "p_value_left_to_right": p_value,
        "significance": sig_stars(p_value),
        "significance_test": (
            "two-sided Welch t-test from summary mean/sample std"),
        "num_runs_for_test": num_runs,
        "lower_is_better": metric["lower_is_better"],
    })
    return left_mean, left_std, right_mean, right_std, pct, p_value


def print_pair_section(title, description, rows, pairs, datasets, config,
                       left_label, right_label, delta_label, num_runs,
                       saved_std_ddof, metrics=None):
    print("\n" + "=" * 130)
    print(title)
    if description:
        print(description)
    print("=" * 130)
    print(f"{'Dataset':10s} {'Family':10s} {'Metric':35s} "
          f"{left_label:>22s} {right_label:>22s} {delta_label:>16s} "
          f"{'p-value':>10s} {'Sig.':>6s}")
    print("-" * 144)

    selected_metrics = METRICS if metrics is None else metrics
    for dataset in datasets:
        for metric in selected_metrics:
            for family, left_alg, right_alg in pairs:
                l_mean, l_std, r_mean, r_std, pct, p_value = append_pair_row(
                    rows, title.lower().replace(" ", "_"), dataset, family,
                    metric, left_label, left_alg, right_label, right_alg,
                    config, num_runs, saved_std_ddof)
                print(f"{dataset:10s} {family:10s} {metric['name'][:35]:35s} "
                      f"{fmt_val(l_mean, l_std):>22s} "
                      f"{fmt_val(r_mean, r_std):>22s} "
                      f"{fmt_pct(pct):>16s} "
                      f"{fmt_pvalue(p_value):>10s} {sig_stars(p_value):>6s}")
        print()


def print_baseline_nofd_section(rows, datasets, config, num_runs,
                                saved_std_ddof):
    print("\n" + "=" * 130)
    print("BASELINE / NoFD / FULL TDGE")
    print("=" * 130)
    print(f"{'Dataset':10s} {'Family':10s} {'Metric':35s} "
          f"{'Baseline':>18s} {'NoFD':>18s} {'Full TDGE':>18s} "
          f"{'D Base->TDGE':>14s} {'p(Base)':>10s} {'Sig.':>6s} "
          f"{'D NoFD->TDGE':>14s} {'p(NoFD)':>10s} {'Sig.':>6s}")
    print("-" * 170)

    for dataset in datasets:
        for metric in METRICS:
            for family, base_alg, nofd_alg, full_alg in ALG_TRIPLETS:
                base_dir = resolve_results_dir(config, dataset, base_alg)
                nofd_dir = resolve_results_dir(config, dataset, nofd_alg)
                full_dir = resolve_results_dir(config, dataset, full_alg)

                b_mean, b_std, b_path = load_final(
                    base_dir, base_alg, metric, num_runs, saved_std_ddof)
                n_mean, n_std, n_path = load_final(
                    nofd_dir, nofd_alg, metric, num_runs, saved_std_ddof)
                f_mean, f_std, f_path = load_final(
                    full_dir, full_alg, metric, num_runs, saved_std_ddof)
                pct_b2f = improvement(b_mean, f_mean, metric["lower_is_better"])
                pct_n2f = improvement(n_mean, f_mean, metric["lower_is_better"])
                p_b2f = pvalue_from_summary(
                    b_mean, b_std, f_mean, f_std,
                    metric["lower_is_better"], num_runs, saved_std_ddof)
                p_n2f = pvalue_from_summary(
                    n_mean, n_std, f_mean, f_std,
                    metric["lower_is_better"], num_runs, saved_std_ddof)

                print(f"{dataset:10s} {family:10s} {metric['name'][:35]:35s} "
                      f"{fmt_val(b_mean, b_std):>18s} "
                      f"{fmt_val(n_mean, n_std):>18s} "
                      f"{fmt_val(f_mean, f_std):>18s} "
                      f"{fmt_pct(pct_b2f):>14s} "
                      f"{fmt_pvalue(p_b2f):>10s} {sig_stars(p_b2f):>6s} "
                      f"{fmt_pct(pct_n2f):>14s} "
                      f"{fmt_pvalue(p_n2f):>10s} {sig_stars(p_n2f):>6s}")

                rows.append({
                    "section": "baseline_nofd_tdge",
                    "dataset": dataset,
                    "family": family,
                    "metric": metric["name"],
                    "baseline_alg": base_alg,
                    "baseline_mean": b_mean,
                    "baseline_std": b_std,
                    "baseline_dir": base_dir,
                    "baseline_path": b_path,
                    "nofd_alg": nofd_alg,
                    "nofd_mean": n_mean,
                    "nofd_std": n_std,
                    "nofd_dir": nofd_dir,
                    "nofd_path": n_path,
                    "full_alg": full_alg,
                    "full_mean": f_mean,
                    "full_std": f_std,
                    "full_dir": full_dir,
                    "full_path": f_path,
                    "impr_base_to_full": pct_b2f,
                    "impr_nofd_to_full": pct_n2f,
                    "p_value_base_to_full": p_b2f,
                    "significance_base_to_full": sig_stars(p_b2f),
                    "p_value_nofd_to_full": p_n2f,
                    "significance_nofd_to_full": sig_stars(p_n2f),
                    "significance_test": (
                        "two-sided Welch t-test from summary mean/sample std"),
                    "num_runs_for_test": num_runs,
                    "lower_is_better": metric["lower_is_better"],
                })
        print()


def collect_missing(rows):
    missing = []
    path_keys = sorted({
        key for row in rows for key in row.keys() if key.endswith("_path")
    })
    for row in rows:
        for key in path_keys:
            path = row.get(key)
            if path and not os.path.exists(path):
                missing.append({
                    "section": row.get("section"),
                    "dataset": row.get("dataset"),
                    "family": row.get("family"),
                    "metric": row.get("metric"),
                    "missing_path": path,
                })
    return pd.DataFrame(missing).drop_duplicates() if missing else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(
        description="Compute final metrics when algorithms come from different training result dirs.")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="JSON config mapping each dataset/algorithm to an npy result dir.")
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Optional dataset subset, e.g. --datasets MovieLens")
    parser.add_argument("--num-runs", type=int, default=DEFAULT_NUM_RUNS,
                        help="Number of runs used to compute saved mean/std curves; used for summary-statistics significance tests.")
    parser.add_argument(
        "--saved-std-ddof", type=int, choices=(0, 1), default=0,
        help=("Degrees of freedom used in the saved std_*.npy files. "
              "The stored values are reported unchanged; population SDs are "
              "converted internally only for Welch tests."))
    args = parser.parse_args()

    config = load_config(args.config)
    datasets = args.datasets or list(config["datasets"].keys())
    rows = []

    print("Using cross-run metric config:", args.config)
    for dataset in datasets:
        dataset_cfg = config["datasets"].get(dataset, {})
        print(f"  {dataset}: default={dataset_cfg.get('default_dir')}")
        for alg, rdir in dataset_cfg.get("alg_dirs", {}).items():
            if rdir:
                print(f"    {alg}: {rdir}")

    print(f"Reported variability: stored standard deviation "
          f"(ddof={args.saved_std_ddof})")
    print(f"Significance test: two-sided Welch t-test from summary mean/sample std "
          f"(n={args.num_runs} runs; * p<0.05, ** p<0.01, *** p<0.001)")

    print_baseline_nofd_section(
        rows, datasets, config, args.num_runs, args.saved_std_ddof)
    print_pair_section(
        "ITEM-SPACE vs FEATURE-SPACE EXPLORATION",
        "  ItemCluster : item-space cluster -> item; Full TDGE : feature cluster -> tag -> item",
        rows, IC_PAIRS, datasets, config,
        "ItemCluster", "Full TDGE", "D IC->TDGE", args.num_runs,
        args.saved_std_ddof)
    print_pair_section(
        "ITEM-LEVEL FILTERING: SCORE-SUM vs FEATURE-FUSION",
        "  Score-Sum : tag-score retrieval; Feature-Fusion : selected-feature embedding retrieval",
        rows, FILTER_PAIRS, datasets, config,
        "Score-Sum", "Feature-Fusion", "D Score->Fusion", args.num_runs,
        args.saved_std_ddof)
    print_pair_section(
        "STRUCTURED BASELINE: H2N-BANDIT vs FULL TDGE",
        "  H2N-Bandit : raw category -> item; Full TDGE : feature cluster -> tag -> item",
        rows, H2N_PAIRS, datasets,
        config, "H2N-Bandit", "Full TDGE", "D H2N->TDGE", args.num_runs,
        args.saved_std_ddof)
    print_pair_section(
        "STRUCTURED BASELINE: COFINEUCB vs TDGE-LINUCB",
        "  CoFineUCB : coarse feature subspace -> full feature space; TDGE-LinUCB : feature cluster -> tag -> item",
        rows, COFINE_PAIRS, datasets,
        config, "CoFineUCB", "TDGE-LinUCB", "D CoFine->TDGE", args.num_runs,
        args.saved_std_ddof)
    movielens_only = [dataset for dataset in datasets if dataset == "MovieLens"]
    print_pair_section(
        "HIERARCHY-DEPTH ABLATION: THREE-LEVEL vs FOUR-LEVEL TDGE",
        "  3L: fine dimension -> feature -> item; 4L: coarse dimension -> fine dimension -> feature -> item",
        rows, DEPTH_PAIRS, movielens_only, config,
        "TDGE (3L)", "TDGE (4L)", "D 3L->4L", args.num_runs,
        args.saved_std_ddof,
        metrics=METRICS[:2])

    os.makedirs(args.save_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.save_dir, "final_metrics_cross_runs.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nRaw cross-run numbers saved to: {csv_path}")

    missing_df = collect_missing(rows)
    if not missing_df.empty:
        missing_path = os.path.join(args.save_dir, "final_metrics_cross_runs_missing.csv")
        missing_df.to_csv(missing_path, index=False)
        print(f"Missing metric files saved to: {missing_path}")


if __name__ == "__main__":
    main()
