import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_CONFIG = "clustering_metric_config.json"
DEFAULT_SAVE_DIR = "./paper_figures"
DEFAULT_NUM_RUNS = 5


ALGORITHMS = [
    ("Linear", "LowD_HieRec_LinearUCB"),
    ("NeuralUCB", "LowD_HieRec_NeuralUCB"),
    ("NeuralTS", "LowD_HieRec_NeuralTS"),
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
        "name": "Cold-Start Average Regret",
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


def load_final(results_dir, alg_key, metric, num_runs, saved_std_ddof=0):
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
    left_sample_std = float(left_std) * scale
    right_sample_std = float(right_std) * scale
    result = stats.ttest_ind_from_stats(
        mean1=float(left_mean),
        std1=left_sample_std,
        nobs1=int(num_runs),
        mean2=float(right_mean),
        std2=right_sample_std,
        nobs2=int(num_runs),
        equal_var=False,
        alternative="two-sided",
    )
    if np.isnan(result.pvalue):
        return None
    return float(result.pvalue)


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


def is_better(left_val, right_val, lower_is_better):
    if left_val is None:
        return False
    if right_val is None:
        return True
    if lower_is_better:
        return left_val < right_val
    return left_val > right_val


def collect_metric_values(method_dirs, family, alg_key, metric, num_runs, saved_std_ddof):
    values = {}
    for method_name, results_dir in method_dirs.items():
        mean, std, path = load_final(results_dir, alg_key, metric, num_runs, saved_std_ddof)
        values[method_name] = {
            "mean": mean,
            "std": std,
            "path": path,
            "dir": results_dir,
        }
    best_method = None
    best_mean = None
    for method_name, value in values.items():
        if is_better(value["mean"], best_mean, metric["lower_is_better"]):
            best_method = method_name
            best_mean = value["mean"]
    return values, best_method


def append_rows(rows, dataset, family, alg_key, metric, method_dirs,
                reference_method, num_runs, saved_std_ddof):
    values, best_method = collect_metric_values(
        method_dirs, family, alg_key, metric, num_runs, saved_std_ddof)
    ref = values.get(reference_method)
    ref_mean = ref["mean"] if ref else None
    ref_std = ref["std"] if ref else None

    for method_name, value in values.items():
        if method_name == reference_method:
            delta = 0.0 if value["mean"] is not None else None
            p_value = None
        else:
            delta = improvement(
                value["mean"], ref_mean, metric["lower_is_better"])
            p_value = pvalue_from_summary(
                value["mean"], value["std"], ref_mean, ref_std,
                metric["lower_is_better"], num_runs, saved_std_ddof)

        rows.append({
            "dataset": dataset,
            "family": family,
            "algorithm": alg_key,
            "metric": metric["name"],
            "clustering_method": method_name,
            "mean": value["mean"],
            "std": value["std"],
            "results_dir": value["dir"],
            "metric_path": value["path"],
            "reference_method": reference_method,
            "reference_mean": ref_mean,
            "reference_std": ref_std,
            "delta_method_to_reference_pct": delta,
            "p_value_method_to_reference": p_value,
            "significance": sig_stars(p_value),
            "best_method": best_method,
            "is_best_method": method_name == best_method,
            "significance_test": "two-sided Welch t-test from summary mean/std",
            "num_runs_for_test": num_runs,
            "lower_is_better": metric["lower_is_better"],
        })


def print_section(rows, dataset, method_dirs, reference_method, active_metrics,
                  algs):
    print("\n" + "=" * 132)
    print(f"CLUSTERING METHOD COMPARISON: {dataset}")
    print(f"Reference method: {reference_method}")
    print("=" * 132)
    print(f"{'Family':10s} {'Metric':35s} {'Method':14s} "
          f"{'Value':>22s} {'Reference':>22s} {'D Method->Ref':>15s} "
          f"{'p-value':>10s} {'Sig.':>6s} {'Best':>10s}")
    print("-" * 145)

    dataset_rows = [row for row in rows if row["dataset"] == dataset]
    for metric in active_metrics:
        for family, _ in algs:
            group = [
                row for row in dataset_rows
                if row["metric"] == metric["name"] and row["family"] == family
            ]
            by_method = {row["clustering_method"]: row for row in group}
            for method_name in method_dirs.keys():
                row = by_method.get(method_name)
                if row is None:
                    continue
                print(f"{family:10s} {metric['name'][:35]:35s} "
                      f"{method_name:14s} "
                      f"{fmt_val(row['mean'], row['std']):>22s} "
                      f"{fmt_val(row['reference_mean'], row['reference_std']):>22s} "
                      f"{fmt_pct(row['delta_method_to_reference_pct']):>15s} "
                      f"{fmt_pvalue(row['p_value_method_to_reference']):>10s} "
                      f"{row['significance']:>6s} "
                      f"{row['best_method']:>10s}")
        print()


def collect_missing(rows):
    missing = []
    for row in rows:
        path = row.get("metric_path")
        if path and not os.path.exists(path):
            missing.append({
                "dataset": row.get("dataset"),
                "family": row.get("family"),
                "algorithm": row.get("algorithm"),
                "metric": row.get("metric"),
                "clustering_method": row.get("clustering_method"),
                "missing_path": path,
            })
    return pd.DataFrame(missing).drop_duplicates() if missing else pd.DataFrame()


def filter_metrics(include_reward):
    if include_reward:
        return METRICS
    return [metric for metric in METRICS if metric["lower_is_better"]]


def main():
    parser = argparse.ArgumentParser(
        description="Compare TDGE final metrics across clustering-method result directories.")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="JSON config mapping each dataset/clustering method to an npy result dir.")
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Optional dataset subset, e.g. --datasets MovieLens LastFM")
    parser.add_argument("--methods", nargs="*", default=None,
                        help="Optional clustering-method subset; keeps config order.")
    parser.add_argument("--reference-method", default=None,
                        help="Reference clustering method. Default: config reference_method, then first method.")
    parser.add_argument("--algorithms", nargs="*", default=None,
                        choices=[family for family, _ in ALGORITHMS],
                        help="Optional backbone subset: Linear NeuralUCB NeuralTS")
    parser.add_argument("--num-runs", type=int, default=DEFAULT_NUM_RUNS,
                        help="Number of seeds used in the saved mean/std curves.")
    parser.add_argument("--saved-std-ddof", type=int, choices=(0, 1), default=0,
                        help="Degrees of freedom used in saved std arrays; stored values are reported unchanged.")
    parser.add_argument("--include-reward", action="store_true",
                        help="Also include Cold-Start Average Reward.")
    args = parser.parse_args()

    config = load_config(args.config)
    datasets = args.datasets or list(config["datasets"].keys())
    algs = [
        (family, alg_key) for family, alg_key in ALGORITHMS
        if args.algorithms is None or family in args.algorithms
    ]

    active_metrics = filter_metrics(args.include_reward)
    rows = []

    print("Using clustering metric config:", args.config)
    print(f"Reported variability: stored standard deviation (ddof={args.saved_std_ddof})")
    print(f"Significance test: two-sided Welch t-test from summary mean/sample std "
          f"(n={args.num_runs}; * p<0.05, ** p<0.01, *** p<0.001)")

    global_reference = args.reference_method or config.get("reference_method")
    for dataset in datasets:
        dataset_cfg = config["datasets"].get(dataset)
        if not dataset_cfg:
            print(f"\nSkipping unknown dataset: {dataset}")
            continue

        method_dirs = dataset_cfg.get("methods", {})
        if args.methods is not None:
            method_dirs = {
                method: path for method, path in method_dirs.items()
                if method in args.methods
            }
        if not method_dirs:
            print(f"\nSkipping {dataset}: no clustering methods configured.")
            continue

        reference_method = global_reference
        if reference_method not in method_dirs:
            reference_method = next(iter(method_dirs))
            print(f"\n{dataset}: requested reference missing; using {reference_method}.")

        print(f"\n{dataset} result dirs:")
        for method_name, results_dir in method_dirs.items():
            marker = " (reference)" if method_name == reference_method else ""
            print(f"  {method_name}: {results_dir}{marker}")

        for metric in active_metrics:
            for family, alg_key in algs:
                append_rows(
                    rows, dataset, family, alg_key, metric, method_dirs,
                    reference_method, args.num_runs, args.saved_std_ddof)
        print_section(
            rows, dataset, method_dirs, reference_method, active_metrics, algs)

    os.makedirs(args.save_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.save_dir, "final_metrics_clustering_methods.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nRaw clustering-method comparison saved to: {csv_path}")

    missing_df = collect_missing(rows)
    if not missing_df.empty:
        missing_path = os.path.join(
            args.save_dir, "final_metrics_clustering_methods_missing.csv")
        missing_df.to_csv(missing_path, index=False)
        print(f"Missing metric files saved to: {missing_path}")


if __name__ == "__main__":
    main()
