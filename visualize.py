
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FAMILY_ORDER = ["Autoformer", "iTransformer", "patchTST", "vanilla_transformer"]
VARIANT_ORDER = ["base", "HC", "mHC"]

# Fixed variant colors used consistently in every plot.
VARIANT_COLORS = {
    "base": "#1f77b4",
    "HC": "#ff7f0e",
    "mHC": "#2ca02c",
}

METRIC_DISPLAY_NAMES = {
    "mse_normalized": "MSE",
    "mae_normalized": "MAE",
    "mse_real": "MSE Real Scale",
    "mae_real": "MAE Real Scale",
    "best_val_loss": "Best Validation Loss",
    "final_val_loss": "Final Validation Loss",
    "val_loss_std": "Validation Loss Std",
    "grad_norm_mean": "Mean Gradient Norm",
    "grad_norm_max": "Max Gradient Norm",
    "grad_norm_std": "Gradient Norm Std",
    "train_seconds": "Training Time",
    "params": "Parameters",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize HC/mHC time-series experiment metrics.")
    parser.add_argument("--summary", type=str, default="summary_all_experiments.csv")
    parser.add_argument("--raw", type=str, default="raw_all_experiments.json")
    parser.add_argument("--out_dir", type=str, default="./viz_results")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--log_for_large_range", action="store_true", default=True)
    return parser.parse_args()


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(text):
    text = str(text)
    text = re.sub(r"[^\w\-.]+", "_", text)
    return text.strip("_")


def metric_title(metric):
    return METRIC_DISPLAY_NAMES.get(str(metric), str(metric))


def variant_color(variant):
    return VARIANT_COLORS.get(str(variant), None)


def infer_variant(model):
    name = str(model).lower()
    if "mhc" in name:
        return "mHC"
    if "hc" in name:
        return "HC"
    return "base"


def normalize_numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_order_columns(df):
    if "variant" not in df.columns:
        df["variant"] = df["model"].apply(infer_variant)

    df["family_order"] = df["family"].apply(lambda x: FAMILY_ORDER.index(x) if x in FAMILY_ORDER else 999)
    df["variant_order_auto"] = df["variant"].apply(lambda x: VARIANT_ORDER.index(x) if x in VARIANT_ORDER else 999)

    if "variant_order" in df.columns:
        df["variant_order_plot"] = df["variant_order"].fillna(df["variant_order_auto"])
    else:
        df["variant_order_plot"] = df["variant_order_auto"]

    return df.sort_values(["experiment", "horizon", "depth", "num_streams", "sinkhorn_iters", "family_order", "variant_order_plot", "model"])


def load_summary(path):
    df = pd.read_csv(path)

    numeric_cols = [
        "variant_order", "horizon", "depth", "num_streams", "sinkhorn_iters", "params",
        "best_epoch", "best_train_loss", "best_val_loss", "final_train_loss", "final_val_loss",
        "val_loss_std", "grad_norm_mean", "grad_norm_max", "grad_norm_std",
        "activation_norm_mean", "activation_norm_std",
        "mse_normalized", "mae_normalized", "mse_real", "mae_real", "train_seconds"
    ]
    df = normalize_numeric(df, numeric_cols)

    if "status" in df.columns:
        df["status"] = df["status"].fillna("unknown")
        ok = df["status"].str.lower().eq("success")
        failed = df.loc[~ok].copy()
        df = df.loc[ok].copy()
    else:
        failed = pd.DataFrame()

    df = add_order_columns(df)
    return df, failed


def load_raw(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def maybe_log_axis(ax, values):
    values = np.asarray([v for v in values if pd.notna(v) and v > 0], dtype=float)
    if len(values) == 0:
        return
    if values.max() / max(values.min(), 1e-12) > 100:
        ax.set_yscale("log")


def grouped_bar(df, metric, out_path, title, ylabel, dpi=160, log_if_needed=True):
    if df.empty or metric not in df.columns:
        return False

    work = df.dropna(subset=[metric]).copy()
    if work.empty:
        return False

    families = [f for f in FAMILY_ORDER if f in set(work["family"])]
    families += [f for f in work["family"].unique() if f not in families]

    variants = [v for v in VARIANT_ORDER if v in set(work["variant"])]
    variants += [v for v in work["variant"].unique() if v not in variants]

    x = np.arange(len(families))
    width = 0.8 / max(len(variants), 1)

    fig, ax = plt.subplots(figsize=(max(9, len(families) * 2.0), 5.8))

    for i, variant in enumerate(variants):
        vals = []
        for family in families:
            sub = work[(work["family"] == family) & (work["variant"] == variant)]
            vals.append(float(sub[metric].mean()) if not sub.empty else np.nan)

        offsets = x - 0.4 + width / 2 + i * width
        ax.bar(offsets, vals, width=width, label=variant, color=variant_color(variant))

    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=25, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Variant")

    if log_if_needed:
        maybe_log_axis(ax, work[metric].values)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def simple_bar(df, metric, out_path, title, ylabel, sort_ascending=True, dpi=160, log_if_needed=True):
    if df.empty or metric not in df.columns:
        return False

    work = df.dropna(subset=[metric]).copy()
    if work.empty:
        return False

    work["label"] = work["family"].astype(str) + " / " + work["variant"].astype(str)
    work = work.sort_values(metric, ascending=sort_ascending)

    fig, ax = plt.subplots(figsize=(max(9, len(work) * 0.65), 5.8))
    ax.bar(np.arange(len(work)), work[metric].values)
    ax.set_xticks(np.arange(len(work)))
    ax.set_xticklabels(work["label"].values, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)

    if log_if_needed:
        maybe_log_axis(ax, work[metric].values)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def scatter_params_vs_metric(df, out_path, metric="mse_normalized", dpi=160):
    if df.empty or metric not in df.columns or "params" not in df.columns:
        return False

    work = df.dropna(subset=["params", metric]).copy()
    if work.empty:
        return False

    fig, ax = plt.subplots(figsize=(8.5, 5.8))

    for variant in VARIANT_ORDER:
        sub = work[work["variant"] == variant]
        if sub.empty:
            continue
        ax.scatter(sub["params"], sub[metric], label=variant, s=70, alpha=0.85, color=variant_color(variant))

        for _, row in sub.iterrows():
            ax.annotate(str(row["family"]), (row["params"], row[metric]), fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.set_title(f"Parameters vs {metric}")
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25)
    ax.legend(title="Variant")
    maybe_log_axis(ax, work["params"].values)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def plot_loss_curves(raw, out_dir, dpi=160):
    if not raw:
        return []

    rows = []
    for key, item in raw.items():
        row = item.get("row", {})
        hist = item.get("history", {})
        train = hist.get("train_loss", []) or []
        val = hist.get("val_loss", []) or []

        max_len = max(len(train), len(val))
        for i in range(max_len):
            rows.append({
                "key": key,
                "epoch": i + 1,
                "experiment": row.get("experiment"),
                "family": row.get("family"),
                "model": row.get("model"),
                "variant": infer_variant(row.get("model")),
                "horizon": row.get("horizon"),
                "depth": row.get("depth"),
                "train_loss": train[i] if i < len(train) else np.nan,
                "val_loss": val[i] if i < len(val) else np.nan,
            })

    hist_df = pd.DataFrame(rows)
    if hist_df.empty:
        return []

    paths = []
    hist_df.to_csv(out_dir / "tables" / "loss_history_long.csv", index=False)

    for (experiment, family), sub in hist_df.groupby(["experiment", "family"], dropna=False):
        fig, ax = plt.subplots(figsize=(9.5, 5.8))

        for model, s in sub.groupby("model"):
            s = s.sort_values("epoch")
            variant = infer_variant(model)
            color = variant_color(variant)
            ax.plot(s["epoch"], s["train_loss"], linestyle="--", marker="o", markersize=3, label=f"{model} train", color=color)
            ax.plot(s["epoch"], s["val_loss"], linestyle="-", marker="o", markersize=3, label=f"{model} val", color=color)

        ax.set_title(f"Train/Val loss by epoch — {experiment} / {family}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)
        maybe_log_axis(ax, pd.concat([sub["train_loss"], sub["val_loss"]]).values)
        ax.legend(fontsize=8, ncols=2)

        fig.tight_layout()
        path = out_dir / "curves" / f"loss_{safe_name(experiment)}_{safe_name(family)}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        paths.append(path)

    return paths


def plot_grad_curves(raw, out_dir, dpi=160):
    if not raw:
        return []

    rows = []
    for key, item in raw.items():
        row = item.get("row", {})
        hist = item.get("history", {})
        grad = hist.get("grad_norm", []) or []
        grad_max = hist.get("grad_norm_max", []) or []

        max_len = max(len(grad), len(grad_max))
        for i in range(max_len):
            rows.append({
                "key": key,
                "epoch": i + 1,
                "experiment": row.get("experiment"),
                "family": row.get("family"),
                "model": row.get("model"),
                "variant": infer_variant(row.get("model")),
                "grad_norm": grad[i] if i < len(grad) else np.nan,
                "grad_norm_max": grad_max[i] if i < len(grad_max) else np.nan,
            })

    grad_df = pd.DataFrame(rows)
    if grad_df.empty:
        return []

    paths = []
    grad_df.to_csv(out_dir / "tables" / "grad_history_long.csv", index=False)

    for (experiment, family), sub in grad_df.groupby(["experiment", "family"], dropna=False):
        fig, ax = plt.subplots(figsize=(9.5, 5.8))

        for model, s in sub.groupby("model"):
            s = s.sort_values("epoch")
            variant = infer_variant(model)
            color = variant_color(variant)
            ax.plot(s["epoch"], s["grad_norm"], linestyle="-", marker="o", markersize=3, label=f"{model} mean", color=color)
            ax.plot(s["epoch"], s["grad_norm_max"], linestyle="--", marker="x", markersize=3, label=f"{model} max", color=color)

        ax.set_title(f"Gradient norm by epoch — {experiment} / {family}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient norm")
        ax.grid(alpha=0.25)
        maybe_log_axis(ax, pd.concat([sub["grad_norm"], sub["grad_norm_max"]]).values)
        ax.legend(fontsize=8, ncols=2)

        fig.tight_layout()
        path = out_dir / "curves" / f"grad_{safe_name(experiment)}_{safe_name(family)}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        paths.append(path)

    return paths


def plot_sweep_lines(df, x_col, metric, out_dir, title_prefix, dpi=160):
    if df.empty or x_col not in df.columns or metric not in df.columns:
        return []

    work = df.dropna(subset=[x_col, metric]).copy()
    if work[x_col].nunique() <= 1:
        return []

    if "variant" not in work.columns:
        work["variant"] = work["model"].apply(infer_variant)

    paths = []
    for experiment, exp_df in work.groupby("experiment"):
        for family, sub in exp_df.groupby("family"):
            fig, ax = plt.subplots(figsize=(8.8, 5.5))

            variants = [v for v in VARIANT_ORDER if v in set(sub["variant"])]
            variants += [v for v in sub["variant"].unique() if v not in variants]

            for variant in variants:
                s = sub[sub["variant"] == variant].sort_values(x_col)
                if s.empty:
                    continue

                # In case the same variant appears more than once for the same x value,
                # aggregate by mean so the sweep curve remains clean.
                s = s.groupby(x_col, as_index=False)[metric].mean()
                ax.plot(
                    s[x_col],
                    s[metric],
                    marker="o",
                    linewidth=2,
                    color=variant_color(variant),
                )

            # Requested style for depth/stream/sinkhorn/horizon sweep plots:
            # title contains only the metric being plotted, no per-image legend.
            ax.set_title(metric_title(metric))
            ax.set_xlabel(x_col)
            ax.set_ylabel(metric_title(metric))
            ax.grid(alpha=0.25)
            maybe_log_axis(ax, sub[metric].values)

            fig.tight_layout()

            # Filename format: ModelFamily_Metric_Experiment.png
            path = out_dir / "sweeps" / f"{safe_name(family)}_{safe_name(metric)}_{safe_name(experiment)}.png"
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            paths.append(path)

    return paths


def flatten_residual_diagnostics(raw):
    rows = []

    for run_key, item in raw.items():
        row = item.get("row", {})
        diag = item.get("residual_diagnostics", {}) or {}

        for module_name, metrics in diag.items():
            out = {
                "run_key": run_key,
                "experiment": row.get("experiment"),
                "family": row.get("family"),
                "model": row.get("model"),
                "variant": infer_variant(row.get("model")),
                "horizon": row.get("horizon"),
                "depth": row.get("depth"),
                "module": module_name,
            }
            out.update(metrics)
            rows.append(out)

    return pd.DataFrame(rows)


def plot_residual_diagnostics(raw, out_dir, dpi=160):
    diag_df = flatten_residual_diagnostics(raw)
    if diag_df.empty:
        return []

    paths = []
    diag_df.to_csv(out_dir / "tables" / "residual_diagnostics_flat.csv", index=False)

    agg_cols = [c for c in ["row_sum_std", "col_sum_std", "spectral_norm", "condition_number", "entropy"] if c in diag_df.columns]
    agg = diag_df.groupby(["experiment", "family", "model", "variant"], as_index=False)[agg_cols].mean()
    agg.to_csv(out_dir / "tables" / "residual_diagnostics_by_model.csv", index=False)

    for metric in agg_cols:
        path = out_dir / "stability" / f"residual_{safe_name(metric)}.png"
        simple_bar(
            agg.rename(columns={metric: f"mean_{metric}"}),
            f"mean_{metric}",
            path,
            f"Residual diagnostics — mean {metric}",
            f"mean {metric}",
            sort_ascending=True,
            dpi=dpi,
            log_if_needed=True,
        )
        paths.append(path)

    return paths


def write_rank_tables(df, out_dir):
    tables_dir = out_dir / "tables"

    metric_cols = [
        "mse_normalized", "mae_normalized", "mse_real", "mae_real",
        "best_val_loss", "final_val_loss", "val_loss_std",
        "grad_norm_mean", "grad_norm_max", "grad_norm_std",
        "train_seconds", "params"
    ]
    existing = [c for c in metric_cols if c in df.columns]

    df.to_csv(tables_dir / "clean_summary_success.csv", index=False)

    if existing:
        rank = df[["experiment", "family", "model", "variant", "horizon", "depth", "num_streams", "sinkhorn_iters"] + existing].copy()
        for metric in existing:
            rank[f"rank_{metric}"] = rank.groupby(["experiment", "horizon", "depth"])[metric].rank(method="min", ascending=True)
        rank.to_csv(tables_dir / "ranked_metrics.csv", index=False)

    benchmark = df[df["experiment"].astype(str).eq("benchmark")].copy()
    if not benchmark.empty:
        for metric in ["mse_normalized", "mae_normalized", "best_val_loss"]:
            if metric in benchmark.columns:
                pivot = benchmark.pivot_table(index="family", columns="variant", values=metric, aggfunc="mean")
                pivot = pivot.reindex(index=[f for f in FAMILY_ORDER if f in pivot.index])
                pivot.to_csv(tables_dir / f"pivot_benchmark_{metric}.csv")

        best = []
        for metric in ["mse_normalized", "mae_normalized", "best_val_loss", "grad_norm_max", "val_loss_std"]:
            if metric in benchmark.columns:
                idx = benchmark.groupby("family")[metric].idxmin()
                tmp = benchmark.loc[idx, ["family", "model", "variant", metric]].copy()
                tmp["metric"] = metric
                tmp = tmp.rename(columns={metric: "value"})
                best.append(tmp)

        if best:
            pd.concat(best, ignore_index=True).to_csv(tables_dir / "best_model_by_family_and_metric.csv", index=False)


def main():
    args = parse_args()

    out_dir = ensure_dir(args.out_dir)
    for sub in ["comparisons", "curves", "sweeps", "stability", "tables"]:
        ensure_dir(out_dir / sub)

    df, failed = load_summary(args.summary)
    raw = load_raw(args.raw)

    write_rank_tables(df, out_dir)

    if not failed.empty:
        failed.to_csv(out_dir / "tables" / "failed_runs.csv", index=False)

    # Main benchmark comparisons.
    benchmark = df[df["experiment"].astype(str).eq("benchmark")].copy()
    base_for_comparison = benchmark if not benchmark.empty else df

    grouped_bar(base_for_comparison, "mse_normalized", out_dir / "comparisons" / "benchmark_mse_normalized_by_family.png",
                "Final Test MSE normalized by family", "MSE normalized", dpi=args.dpi)

    grouped_bar(base_for_comparison, "mae_normalized", out_dir / "comparisons" / "benchmark_mae_normalized_by_family.png",
                "Final Test MAE normalized by family", "MAE normalized", dpi=args.dpi)

    grouped_bar(base_for_comparison, "best_val_loss", out_dir / "comparisons" / "benchmark_best_val_loss_by_family.png",
                "Best validation loss by family", "Best val loss", dpi=args.dpi)

    grouped_bar(base_for_comparison, "mse_real", out_dir / "comparisons" / "benchmark_mse_real_by_family.png",
                "Final Test MSE real scale by family", "MSE real scale", dpi=args.dpi)

    grouped_bar(base_for_comparison, "mae_real", out_dir / "comparisons" / "benchmark_mae_real_by_family.png",
                "Final Test MAE real scale by family", "MAE real scale", dpi=args.dpi)

    # Stability and efficiency.
    simple_bar(base_for_comparison, "val_loss_std", out_dir / "stability" / "val_loss_std_ranked.png",
               "Validation-loss variability, lower is better", "std(val_loss)", dpi=args.dpi)

    simple_bar(base_for_comparison, "grad_norm_max", out_dir / "stability" / "grad_norm_max_ranked.png",
               "Max gradient norm, lower is usually more stable", "max grad norm", dpi=args.dpi)

    simple_bar(base_for_comparison, "grad_norm_std", out_dir / "stability" / "grad_norm_std_ranked.png",
               "Gradient norm variability, lower is usually more stable", "std(grad_norm)", dpi=args.dpi)

    simple_bar(base_for_comparison, "train_seconds", out_dir / "comparisons" / "train_time_ranked.png",
               "Training time by model", "seconds", dpi=args.dpi)

    scatter_params_vs_metric(base_for_comparison, out_dir / "comparisons" / "params_vs_mse_normalized.png", "mse_normalized", dpi=args.dpi)

    # Per-epoch curves.
    plot_loss_curves(raw, out_dir, dpi=args.dpi)
    plot_grad_curves(raw, out_dir, dpi=args.dpi)

    # Sweep plots.
    for metric in ["mse_normalized", "mae_normalized", "best_val_loss", "grad_norm_max"]:
        plot_sweep_lines(df[df["experiment"].astype(str).eq("horizon")], "horizon", metric, out_dir, "Horizon sweep", dpi=args.dpi)
        plot_sweep_lines(df[df["experiment"].astype(str).eq("depth")], "depth", metric, out_dir, "Depth scaling", dpi=args.dpi)
        plot_sweep_lines(df[df["experiment"].astype(str).eq("streams")], "num_streams", metric, out_dir, "Stream ablation", dpi=args.dpi)
        plot_sweep_lines(df[df["experiment"].astype(str).eq("sinkhorn")], "sinkhorn_iters", metric, out_dir, "Sinkhorn ablation", dpi=args.dpi)

    # Residual matrix diagnostics.
    plot_residual_diagnostics(raw, out_dir, dpi=args.dpi)

    # Simple report.
    report_lines = []
    report_lines.append("# Visualization report\n")
    report_lines.append(f"- Summary file: `{args.summary}`")
    report_lines.append(f"- Raw file: `{args.raw}`")
    report_lines.append(f"- Successful runs visualized: {len(df)}")
    report_lines.append(f"- Failed runs: {len(failed)}")
    report_lines.append("")
    report_lines.append("## Key output folders")
    report_lines.append("- `comparisons/`: benchmark MSE/MAE/best-val/time/params plots")
    report_lines.append("- `curves/`: train/val loss curves and gradient curves")
    report_lines.append("- `sweeps/`: horizon/depth/stream/sinkhorn line plots when available")
    report_lines.append("- `stability/`: val-loss std, grad norm, residual diagnostics")
    report_lines.append("- `tables/`: cleaned summary, pivots, ranks, residual diagnostics CSVs")
    (out_dir / "README_visualization.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Done. Visualizations saved to: {out_dir}")


if __name__ == "__main__":
    main()
