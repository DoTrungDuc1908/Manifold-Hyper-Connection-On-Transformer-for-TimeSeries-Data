import argparse
import csv
import copy
import importlib
import json
import random
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_provider.data_loader import get_data_loaders
from exp.train_ts import Exp_Main

try:
    from utils.trackers import StabilityTracker
except Exception:
    StabilityTracker = None


MODEL_GROUPS = OrderedDict({
    "Autoformer": ["Autoformer", "HC_Autoformer", "mHC_Autoformer"],
    "iTransformer": ["iTransformer", "HC_iTransformer", "mHC_iTransformer"],
    "vanilla_transformer": ["vanilla_transformer", "HC_vanilla_transformer", "mHC_vanilla_transformer"],
    "patchTST": ["patchtst", "HC_patchTST", "mHC_patchtst"],
})


def parse_int_list(text):
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="HC/mHC time-series experiment pipeline")

    # Data
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--data", type=str, default="custom")
    parser.add_argument("--root_path", type=str, default=None)
    parser.add_argument("--features", type=str, default="M", choices=["S", "M", "MS"])
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--num_workers", type=int, default=0)

    # Task / sequence
    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=96)

    # Model
    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--dec_in", type=int, default=7)
    parser.add_argument("--c_out", type=int, default=7)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)

    # HC / mHC
    parser.add_argument("--num_streams", type=int, default=4)
    parser.add_argument("--sinkhorn_iters", type=int, default=10)
    parser.add_argument("--resid_dropout", type=float, default=0.05)
    parser.add_argument("--ffn_dropout", type=float, default=0.1)
    parser.add_argument("--stream_init", type=str, default="embed", choices=["embed", "repeat"])
    parser.add_argument("--stream_collapse", type=str, default="learnable", choices=["learnable", "mean"])

    # Flags
    parser.add_argument("--output_attention", action="store_true", default=False)
    parser.add_argument("--no_skip", action="store_true", default=False)
    parser.add_argument("--fuse_decoder", action="store_true", default=False)
    parser.add_argument("--decoder_type", type=str, default="conv2d")
    parser.add_argument("--no_zero_norm", action="store_true", default=False)
    parser.add_argument("--use_norm", type=int, default=1)

    # Optimization
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--min_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--disable_early_stopping", action="store_true", default=False)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_activation_norm", action="store_true", default=False)

    # Experiment sweeps
    parser.add_argument("--experiment", type=str, default="benchmark",
                        choices=["benchmark", "diagnostic", "horizon", "depth", "streams", "sinkhorn", "all_sweeps"])
    parser.add_argument("--horizons", type=str, default="24,48,96,192,336,720")
    parser.add_argument("--depths", type=str, default="2,4,6,8")
    parser.add_argument("--stream_values", type=str, default="1,2,4,8")
    parser.add_argument("--sinkhorn_values", type=str, default="1,3,5,10,20")

    # Management
    parser.add_argument("--model_name", type=str, default="all")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--save_dir", type=str, default="./eval_results")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--continue_on_error", action="store_true", default=False)

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_model_name(name):
    return name.lower().replace("-", "_")


def flatten_model_groups():
    models = []
    for names in MODEL_GROUPS.values():
        models.extend(names)
    return models


def get_family_name(model_name):
    normalized = normalize_model_name(model_name)
    for family, names in MODEL_GROUPS.items():
        if normalized == normalize_model_name(family):
            return family
        for name in names:
            if normalized == normalize_model_name(name):
                return family
    return "custom"


def get_variant_order(model_name):
    family = get_family_name(model_name)
    if family == "custom":
        return 999
    for idx, name in enumerate(MODEL_GROUPS[family]):
        if normalize_model_name(name) == normalize_model_name(model_name):
            return idx
    return 999


def resolve_models_to_run(model_name):
    normalized = normalize_model_name(model_name)
    if normalized == "all":
        return flatten_model_groups()
    for family, names in MODEL_GROUPS.items():
        if normalized == normalize_model_name(family):
            return names
    return [model_name]


def filter_models_for_experiment(models, experiment):
    if experiment == "streams":
        return [m for m in models if "hc" in normalize_model_name(m)]
    if experiment == "sinkhorn":
        return [m for m in models if "mhc" in normalize_model_name(m)]
    return models


def list_model_files(models_dir):
    if not models_dir.exists():
        raise FileNotFoundError(f"models_dir does not exist: {models_dir}")
    return [file.stem for file in models_dir.glob("*.py") if file.name != "__init__.py"]


def resolve_module_name(model_name, models_dir):
    available_files = list_model_files(models_dir)
    for file_name in available_files:
        if normalize_model_name(file_name) == normalize_model_name(model_name):
            return file_name
    raise ValueError(f"Could not find model file for '{model_name}'. Available model files: {available_files}")


def get_model_instance(model_name, args):
    project_root = Path(__file__).resolve().parent
    models_dir = Path(args.models_dir)
    if not models_dir.is_absolute():
        models_dir = project_root / models_dir
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    target_module_name = resolve_module_name(model_name, models_dir)
    module = importlib.import_module(f"models.{target_module_name}")
    if not hasattr(module, "Model"):
        raise AttributeError(f"models/{target_module_name}.py does not contain class Model.")

    model = module.Model(configs=args)
    print(f"Model loaded: {target_module_name}")
    return model, target_module_name


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_experiment_dirs(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.exp_name or f"{args.experiment}_{args.task_name}_seq{args.seq_len}_{timestamp}"
    root = Path(args.save_dir) / exp_name
    dirs = {
        "root": root,
        "weights": root / "saved_weights",
        "plots": root / "plots",
        "logs": root / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def write_csv(path, rows):
    fieldnames = [
        "experiment", "family", "variant_order", "model", "status", "error",
        "horizon", "depth", "num_streams", "sinkhorn_iters",
        "stream_init", "stream_collapse",
        "params", "best_epoch", "best_train_loss", "best_val_loss",
        "final_train_loss", "final_val_loss", "val_loss_std",
        "grad_norm_mean", "grad_norm_max", "grad_norm_std",
        "activation_norm_mean", "activation_norm_std",
        "mse_normalized", "mae_normalized", "mse_real", "mae_real",
        "train_seconds", "best_weight_path", "loaded_best_before_test",
    ]

    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sort_rows(rows):
    family_order = {family: idx for idx, family in enumerate(MODEL_GROUPS.keys())}
    return sorted(rows, key=lambda r: (
        str(r.get("experiment", "")),
        int(r.get("horizon", 0) or 0),
        int(r.get("depth", 0) or 0),
        int(r.get("num_streams", 0) or 0),
        int(r.get("sinkhorn_iters", 0) or 0),
        family_order.get(r.get("family", ""), 999),
        int(r.get("variant_order", 999) or 999),
        normalize_model_name(r.get("model", "")),
    ))


def get_data(args):
    return get_data_loaders(
        file_path=args.data_path,
        root_path=args.root_path,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        features=args.features,
        target=args.target,
        scale=bool(args.scale),
        embed=args.embed,
        freq=args.freq,
        data_name=args.data,
        num_workers=args.num_workers,
    )


def load_best_if_exists(exp, weight_path, device):
    if weight_path.exists():
        state_dict = torch.load(weight_path, map_location=device)
        exp.model.load_state_dict(state_dict)
        print(f"Loaded best checkpoint before test: {weight_path}")
        return True
    print(f"Best checkpoint not found: {weight_path}. Testing current weights.")
    return False


def summarize_history(values):
    if not values:
        return "", "", ""
    return min(values), values[-1], float(np.std(values))


def build_row(args, experiment_name, family, variant_order, model_name, params, exp, mse, mae, weight_path, loaded_best, status="success", error=""):
    train_best, train_final, _ = summarize_history(getattr(exp, "train_loss_history", []))
    val_best, val_final, val_std = summarize_history(getattr(exp, "val_loss_history", []))

    grad_hist = getattr(exp, "grad_norm_history", [])
    grad_max_hist = getattr(exp, "grad_norm_max_history", [])
    act_hist = getattr(exp, "activation_norm_history", [])
    test_metrics = getattr(exp, "test_metrics", {})

    return {
        "experiment": experiment_name,
        "family": family,
        "variant_order": variant_order,
        "model": model_name,
        "status": status,
        "error": error,
        "horizon": args.pred_len,
        "depth": args.e_layers,
        "num_streams": getattr(args, "num_streams", ""),
        "sinkhorn_iters": getattr(args, "sinkhorn_iters", ""),
        "stream_init": getattr(args, "stream_init", ""),
        "stream_collapse": getattr(args, "stream_collapse", ""),
        "params": params,
        "best_epoch": getattr(exp, "best_epoch", ""),
        "best_train_loss": train_best,
        "best_val_loss": val_best,
        "final_train_loss": train_final,
        "final_val_loss": val_final,
        "val_loss_std": val_std,
        "grad_norm_mean": float(np.mean(grad_hist)) if grad_hist else "",
        "grad_norm_max": float(np.max(grad_max_hist)) if grad_max_hist else "",
        "grad_norm_std": float(np.std(grad_hist)) if grad_hist else "",
        "activation_norm_mean": float(np.mean(act_hist)) if act_hist else "",
        "activation_norm_std": float(np.std(act_hist)) if act_hist else "",
        "mse_normalized": test_metrics.get("mse_normalized", mse),
        "mae_normalized": test_metrics.get("mae_normalized", mae),
        "mse_real": test_metrics.get("mse_real", ""),
        "mae_real": test_metrics.get("mae_real", ""),
        "train_seconds": float(np.sum(getattr(exp, "epoch_time_history", []))),
        "best_weight_path": str(weight_path),
        "loaded_best_before_test": loaded_best,
    }


def run_single(args, experiment_name, model_name, dirs, device):
    family = get_family_name(model_name)
    variant_order = get_variant_order(model_name)
    weight_dir = dirs["weights"] / experiment_name / family
    weight_dir.mkdir(parents=True, exist_ok=True)

    run_tag = f"pred{args.pred_len}_depth{args.e_layers}_s{args.num_streams}_sk{args.sinkhorn_iters}"
    weight_path = weight_dir / f"{model_name}_{run_tag}_best.pth"

    print("\n" + "=" * 100)
    print(f"Experiment: {experiment_name} | Model: {model_name} | pred_len={args.pred_len} | depth={args.e_layers}")
    print("=" * 100)

    train_loader, val_loader, test_loader, scaler = get_data(args)
    model, resolved_name = get_model_instance(model_name, args)
    params = count_parameters(model)
    tracker = StabilityTracker() if StabilityTracker is not None else None

    exp = Exp_Main(
        args=args,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scaler=scaler,
        device=device,
        learning_rate=args.learning_rate,
        tracker=tracker,
    )

    exp.train(
        epochs=args.epochs,
        patience=args.patience,
        weight_save_path=str(weight_path),
        min_epochs=args.min_epochs,
        disable_early_stopping=args.disable_early_stopping,
    )

    loaded_best = load_best_if_exists(exp, weight_path, device)
    mse, mae = exp.test(return_real_metrics=False)

    row = build_row(args, experiment_name, family, variant_order, resolved_name, params, exp, mse, mae, weight_path, loaded_best)

    raw = {
        "config": vars(args).copy(),
        "row": row,
        "history": {
            "train_loss": getattr(exp, "train_loss_history", []),
            "val_loss": getattr(exp, "val_loss_history", []),
            "grad_norm": getattr(exp, "grad_norm_history", []),
            "grad_norm_max": getattr(exp, "grad_norm_max_history", []),
            "activation_norm": getattr(exp, "activation_norm_history", []),
            "epoch_time": getattr(exp, "epoch_time_history", []),
        },
        "test_metrics": getattr(exp, "test_metrics", {}),
        "residual_diagnostics": getattr(exp, "residual_diagnostics", {}),
    }

    del exp
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row, raw


def build_runs(args):
    base_models = resolve_models_to_run(args.model_name)
    runs = []

    def add_runs(exp_name, override_list):
        models = filter_models_for_experiment(base_models, exp_name)
        for overrides in override_list:
            for model_name in models:
                run_args = copy.deepcopy(args)
                for key, value in overrides.items():
                    setattr(run_args, key, value)
                if exp_name == "diagnostic":
                    run_args.disable_early_stopping = True
                    run_args.min_epochs = max(run_args.min_epochs, run_args.epochs)
                    run_args.patience = max(run_args.patience, 999)
                runs.append((exp_name, model_name, run_args))

    if args.experiment in ["benchmark", "all_sweeps"]:
        add_runs("benchmark", [{}])

    if args.experiment in ["diagnostic", "all_sweeps"]:
        add_runs("diagnostic", [{}])

    if args.experiment in ["horizon", "all_sweeps"]:
        add_runs("horizon", [{"pred_len": h} for h in parse_int_list(args.horizons)])

    if args.experiment in ["depth", "all_sweeps"]:
        add_runs("depth", [{"e_layers": d} for d in parse_int_list(args.depths)])

    if args.experiment in ["streams", "all_sweeps"]:
        add_runs("streams", [{"num_streams": s} for s in parse_int_list(args.stream_values)])

    if args.experiment in ["sinkhorn", "all_sweeps"]:
        add_runs("sinkhorn", [{"sinkhorn_iters": k} for k in parse_int_list(args.sinkhorn_values)])

    return runs


def plot_summary(dirs, rows):
    valid = [r for r in rows if r["status"] == "success"]
    if not valid:
        return

    labels = [f'{r["model"]}\\nP{r["horizon"]}/D{r["depth"]}' for r in valid]
    mse = [float(r["mse_normalized"]) for r in valid]
    mae = [float(r["mae_normalized"]) for r in valid]

    for name, values, ylabel in [("mse_comparison.png", mse, "MSE normalized"), ("mae_comparison.png", mae, "MAE normalized")]:
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.55), 6))
        ax.bar(x, values)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(ylabel)
        plt.tight_layout()
        plt.savefig(dirs["plots"] / name)
        plt.close()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dirs = make_experiment_dirs(args)

    write_json(dirs["logs"] / "base_config.json", vars(args))
    print(f"Device: {device}")
    print(f"Experiment directory: {dirs['root']}")

    rows = []
    raw = OrderedDict()
    runs = build_runs(args)

    for idx, (experiment_name, model_name, run_args) in enumerate(runs, start=1):
        key = f"{idx:04d}_{experiment_name}_{model_name}_pred{run_args.pred_len}_depth{run_args.e_layers}_s{run_args.num_streams}_sk{run_args.sinkhorn_iters}"

        try:
            row, raw_item = run_single(run_args, experiment_name, model_name, dirs, device)
            rows.append(row)
            raw[key] = raw_item

        except Exception as exc:
            family = get_family_name(model_name)
            row = {
                "experiment": experiment_name,
                "family": family,
                "variant_order": get_variant_order(model_name),
                "model": model_name,
                "status": "failed",
                "error": repr(exc),
                "horizon": run_args.pred_len,
                "depth": run_args.e_layers,
                "num_streams": getattr(run_args, "num_streams", ""),
                "sinkhorn_iters": getattr(run_args, "sinkhorn_iters", ""),
                "stream_init": getattr(run_args, "stream_init", ""),
                "stream_collapse": getattr(run_args, "stream_collapse", ""),
            }
            rows.append(row)
            raw[key] = {"config": vars(run_args).copy(), "row": row}
            print(f"Failed: {key} | {repr(exc)}")
            if not args.continue_on_error:
                raise

        rows_sorted = sort_rows(rows)
        write_csv(dirs["logs"] / "summary_all_experiments.csv", rows_sorted)
        write_json(dirs["logs"] / "raw_all_experiments.json", raw)

    rows_sorted = sort_rows(rows)
    write_csv(dirs["logs"] / "summary_all_experiments.csv", rows_sorted)
    write_json(dirs["logs"] / "raw_all_experiments.json", raw)
    plot_summary(dirs, rows_sorted)

    print("\nFinished.")
    print(f"Summary CSV: {dirs['logs'] / 'summary_all_experiments.csv'}")
    print(f"Raw JSON: {dirs['logs'] / 'raw_all_experiments.json'}")
    print(f"Weights: {dirs['weights']}")


if __name__ == "__main__":
    main()
