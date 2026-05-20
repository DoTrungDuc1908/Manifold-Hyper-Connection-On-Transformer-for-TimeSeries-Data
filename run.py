import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd


def quote_cmd(cmd):
    return " ".join(shlex.quote(str(x)) for x in cmd)


def find_finance_dataset():
    """
    Priority:
        1. ./dataset/finance/stock_panel.csv
        2. ./dataset/exchange_rate/exchange_rate.csv

    exchange_rate is used as a finance-like benchmark if your own stock/crypto file
    has not been added yet.
    """
    candidates = [
        Path("./dataset/finance/stock_panel.csv"),
        Path("./dataset/exchange_rate/exchange_rate.csv"),
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Không tìm thấy finance dataset. Cần một trong hai file:\n"
        "  - ./dataset/finance/stock_panel.csv\n"
        "  - ./dataset/exchange_rate/exchange_rate.csv"
    )


def non_date_columns(df):
    return [
        c for c in df.columns
        if c.lower() not in ["date", "timestamp", "time"]
    ]


def infer_target_and_dims(csv_path):
    df = pd.read_csv(csv_path, nrows=5)
    data_cols = non_date_columns(df)

    if len(data_cols) == 0:
        raise ValueError(f"Không tìm thấy cột dữ liệu trong {csv_path}")

    target_candidates = [
        "close",
        "Close",
        "adj_close",
        "Adj Close",
        "OT",
        data_cols[-1],
    ]

    target = None
    for cand in target_candidates:
        if cand in df.columns:
            target = cand
            break

    if target is None:
        target = data_cols[-1]

    # Smoke test dùng multivariate forecasting nếu có nhiều cột.
    features = "M"
    enc_in = len(data_cols)
    dec_in = len(data_cols)
    c_out = len(data_cols)

    return target, features, enc_in, dec_in, c_out


def infer_freq(csv_path):
    name = str(csv_path).lower()

    if "exchange_rate" in name or "finance" in name or "stock" in name or "crypto" in name:
        return "d"

    return "d"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke test: run all model variants on one finance dataset with tiny config."
    )

    parser.add_argument("--main_file", type=str, default="main.py")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--dataset", type=str, default=None,
                        help="Optional explicit CSV path. If omitted, script uses stock_panel.csv or exchange_rate.csv.")
    parser.add_argument("--exp_name", type=str, default="SMOKE_finance_all_models_tiny")
    parser.add_argument("--save_dir", type=str, default="./eval_results")
    parser.add_argument("--dry_run", action="store_true", default=False)

    # Tiny model config for fast code checking.
    parser.add_argument("--seq_len", type=int, default=36)
    parser.add_argument("--label_len", type=int, default=18)
    parser.add_argument("--pred_len", type=int, default=5)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--e_layers", type=int, default=1)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--num_streams", type=int, default=2)
    parser.add_argument("--sinkhorn_iters", type=int, default=3)

    # Fast train config.
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--min_epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_args()

    csv_path = Path(args.dataset) if args.dataset is not None else find_finance_dataset()
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset không tồn tại: {csv_path}")

    target, features, enc_in, dec_in, c_out = infer_target_and_dims(csv_path)
    freq = infer_freq(csv_path)

    cmd = [
        args.python,
        args.main_file,

        "--data_path", str(csv_path),
        "--data", "custom",
        "--model_name", "all",
        "--experiment", "benchmark",

        "--seq_len", args.seq_len,
        "--label_len", args.label_len,
        "--pred_len", args.pred_len,
        "--features", features,
        "--target", target,
        "--enc_in", enc_in,
        "--dec_in", dec_in,
        "--c_out", c_out,

        "--d_model", args.d_model,
        "--n_heads", args.n_heads,
        "--e_layers", args.e_layers,
        "--d_layers", args.d_layers,
        "--d_ff", args.d_ff,
        "--dropout", 0.1,
        "--resid_dropout", 0.05,
        "--ffn_dropout", 0.1,
        "--num_streams", args.num_streams,
        "--sinkhorn_iters", args.sinkhorn_iters,
        "--factor", 1,
        "--activation", "gelu",
        "--moving_avg", 7,
        "--patch_len", 8,
        "--stride", 4,

        "--embed", "timeF",
        "--freq", freq,
        "--scale", 1,

        "--batch_size", args.batch_size,
        "--learning_rate", args.learning_rate,
        "--epochs", args.epochs,
        "--min_epochs", args.min_epochs,
        "--patience", args.patience,
        "--grad_clip", 1.0,

        "--save_dir", args.save_dir,
        "--exp_name", args.exp_name,
        "--continue_on_error",
    ]

    print("=" * 100)
    print("Smoke test finance dataset")
    print("=" * 100)
    print(f"Dataset:  {csv_path}")
    print(f"Target:   {target}")
    print(f"Features: {features}")
    print(f"Dims:     enc_in={enc_in}, dec_in={dec_in}, c_out={c_out}")
    print(f"Freq:     {freq}")
    print("\nCommand:")
    print(quote_cmd(cmd))
    print("=" * 100)

    if args.dry_run:
        print("Dry run only. Không chạy training.")
        return

    cmd = [str(x) for x in cmd]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    result_dir = Path(args.save_dir) / args.exp_name
    summary = result_dir / "logs" / "summary_all_experiments.csv"
    raw_json = result_dir / "logs" / "raw_all_experiments.json"

    print("\nSmoke test finished.")
    print(f"Result dir: {result_dir}")

    if summary.exists():
        print(f"Summary CSV: {summary}")
    else:
        print(f"Không thấy summary CSV: {summary}")

    if raw_json.exists():
        print(f"Raw JSON: {raw_json}")
    else:
        print(f"Không thấy raw JSON: {raw_json}")

    print("\nNên mở summary_all_experiments.csv để kiểm tra:")
    print("  - status có success/failed không")
    print("  - mse_normalized / mae_normalized có giá trị hợp lý không")
    print("  - best_weight_path có file .pth không")


if __name__ == "__main__":
    main()
