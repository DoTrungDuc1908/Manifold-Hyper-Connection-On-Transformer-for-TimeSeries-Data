import os
import warnings
from typing import Optional, Sequence, Tuple, List

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


def time_features(dates, freq="h"):
    """
    Lightweight replacement for utils.timefeatures.time_features.

    Return shape:
        [num_time_features, length]

    Used when embed='timeF'. Values are roughly centered to [-0.5, 0.5].
    """
    dates = pd.to_datetime(dates)
    if not isinstance(dates, pd.Series):
        dates = pd.Series(dates)

    features = []

    if freq in ["m", "month"]:
        features.append(dates.dt.month.values / 12.0 - 0.5)

    elif freq in ["w", "week"]:
        features.append(dates.dt.month.values / 12.0 - 0.5)
        features.append(dates.dt.weekday.values / 6.0 - 0.5)

    elif freq in ["d", "b", "day"]:
        features.append(dates.dt.month.values / 12.0 - 0.5)
        features.append(dates.dt.day.values / 31.0 - 0.5)
        features.append(dates.dt.weekday.values / 6.0 - 0.5)

    elif freq in ["h", "hour"]:
        features.append(dates.dt.month.values / 12.0 - 0.5)
        features.append(dates.dt.day.values / 31.0 - 0.5)
        features.append(dates.dt.weekday.values / 6.0 - 0.5)
        features.append(dates.dt.hour.values / 23.0 - 0.5)

    elif freq in ["t", "min", "15min", "minute"]:
        features.append(dates.dt.month.values / 12.0 - 0.5)
        features.append(dates.dt.day.values / 31.0 - 0.5)
        features.append(dates.dt.weekday.values / 6.0 - 0.5)
        features.append(dates.dt.hour.values / 23.0 - 0.5)
        features.append(dates.dt.minute.values / 59.0 - 0.5)

    elif freq in ["s", "sec", "second"]:
        features.append(dates.dt.month.values / 12.0 - 0.5)
        features.append(dates.dt.day.values / 31.0 - 0.5)
        features.append(dates.dt.weekday.values / 6.0 - 0.5)
        features.append(dates.dt.hour.values / 23.0 - 0.5)
        features.append(dates.dt.minute.values / 59.0 - 0.5)
        features.append(dates.dt.second.values / 59.0 - 0.5)

    else:
        features.append(dates.dt.month.values / 12.0 - 0.5)
        features.append(dates.dt.day.values / 31.0 - 0.5)
        features.append(dates.dt.weekday.values / 6.0 - 0.5)
        features.append(dates.dt.hour.values / 23.0 - 0.5)

    return np.vstack(features)


def _infer_freq_for_pandas(freq):
    freq_map = {
        "h": "H",
        "t": "15min",
        "min": "T",
        "15min": "15min",
        "s": "S",
        "d": "D",
        "b": "B",
        "w": "W",
        "m": "M",
        "a": "A",
    }
    return freq_map.get(freq, freq)


def _make_calendar_features(date_values, freq="h"):
    df_stamp = pd.DataFrame({"date": pd.to_datetime(date_values)})
    df_stamp["month"] = df_stamp.date.apply(lambda row: row.month)
    df_stamp["day"] = df_stamp.date.apply(lambda row: row.day)
    df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday())
    df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour)

    if freq in ["t", "min", "15min", "minute"]:
        df_stamp["minute"] = df_stamp.date.apply(lambda row: row.minute)
        df_stamp["minute"] = df_stamp.minute.map(lambda x: x // 15)

    return df_stamp.drop(["date"], axis=1).values


class ForecastDataset(Dataset):
    """
    Unified forecasting dataset for Autoformer, vanilla Transformer, iTransformer, and PatchTST.

    Returns:
        seq_x:      [seq_len, num_input_features]
        seq_y:      [label_len + pred_len, num_output_features]
        seq_x_mark: [seq_len, time_feature_dim]
        seq_y_mark: [label_len + pred_len, time_feature_dim]

    This matches the official Autoformer/iTransformer/PatchTST sliding-window convention.
    """

    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="M",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
        data_name="custom",
        split_ratios=(0.7, 0.1, 0.2),
        cols: Optional[Sequence[str]] = None,
    ):
        assert flag in ["train", "val", "test"]
        self.flag = flag
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]

        if size is None:
            self.seq_len, self.label_len, self.pred_len = 96, 48, 96
        else:
            self.seq_len, self.label_len, self.pred_len = size

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.data_name = data_name
        self.split_ratios = split_ratios
        self.cols = list(cols) if cols is not None else None
        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()

    def _get_borders(self, df_raw):
        name = self.data_name.lower()
        n = len(df_raw)

        if name in ["etth1", "etth2"]:
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

        elif name in ["ettm1", "ettm2"]:
            border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]

        else:
            train_ratio, val_ratio, test_ratio = self.split_ratios
            num_train = int(n * train_ratio)
            num_test = int(n * test_ratio)
            num_val = n - num_train - num_test

            border1s = [0, num_train - self.seq_len, n - num_test - self.seq_len]
            border2s = [num_train, num_train + num_val, n]

        border1 = max(0, border1s[self.set_type])
        border2 = min(n, border2s[self.set_type])
        return border1, border2, border1s, border2s

    def _select_columns(self, df_raw):
        if "date" not in df_raw.columns:
            df_raw = df_raw.copy()
            df_raw.insert(0, "date", pd.date_range("2000-01-01", periods=len(df_raw), freq=_infer_freq_for_pandas(self.freq)))

        if self.cols is not None:
            cols = self.cols.copy()
            if self.target in cols:
                cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove("date")
            if self.target in cols:
                cols.remove(self.target)

        # Keep target as the last column, following THUML data loaders.
        if self.target in df_raw.columns:
            df_raw = df_raw[["date"] + cols + [self.target]]
        else:
            df_raw = df_raw[["date"] + cols]

        if self.features in ["M", "MS"]:
            df_data = df_raw[df_raw.columns[1:]]
        elif self.features == "S":
            if self.target not in df_raw.columns:
                raise ValueError(f"target='{self.target}' is not found in {self.data_path}.")
            df_data = df_raw[[self.target]]
        else:
            raise ValueError("features must be one of ['S', 'M', 'MS'].")

        return df_raw, df_data

    def _build_time_marks(self, df_raw, border1, border2):
        df_stamp = df_raw[["date"]][border1:border2].copy()
        df_stamp["date"] = pd.to_datetime(df_stamp.date)

        if self.timeenc == 0:
            data_stamp = _make_calendar_features(df_stamp.date.values, freq=self.freq)
        else:
            data_stamp = time_features(df_stamp["date"].values, freq=self.freq).transpose(1, 0)

        return data_stamp

    def __read_data__(self):
        self.scaler = StandardScaler()

        file_path = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_csv(file_path)
        df_raw, df_data = self._select_columns(df_raw)

        border1, border2, border1s, border2s = self._get_borders(df_raw)

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        data_stamp = self._build_time_marks(df_raw, border1, border2)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return (
            torch.tensor(seq_x, dtype=torch.float32),
            torch.tensor(seq_y, dtype=torch.float32),
            torch.tensor(seq_x_mark, dtype=torch.float32),
            torch.tensor(seq_y_mark, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    """
    Prediction dataset matching the Autoformer-style pred loader.
    """

    def __init__(
        self,
        root_path,
        flag="pred",
        size=None,
        features="M",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        inverse=False,
        timeenc=0,
        freq="h",
        cols: Optional[Sequence[str]] = None,
    ):
        assert flag == "pred"

        if size is None:
            self.seq_len, self.label_len, self.pred_len = 96, 48, 96
        else:
            self.seq_len, self.label_len, self.pred_len = size

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = list(cols) if cols is not None else None
        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        if "date" not in df_raw.columns:
            df_raw.insert(0, "date", pd.date_range("2000-01-01", periods=len(df_raw), freq=_infer_freq_for_pandas(self.freq)))

        if self.cols is not None:
            cols = self.cols.copy()
            if self.target in cols:
                cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove("date")
            if self.target in cols:
                cols.remove(self.target)

        df_raw = df_raw[["date"] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features in ["M", "MS"]:
            df_data = df_raw[df_raw.columns[1:]]
        elif self.features == "S":
            df_data = df_raw[[self.target]]
        else:
            raise ValueError("features must be one of ['S', 'M', 'MS'].")

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[["date"]][border1:border2].copy()
        tmp_stamp["date"] = pd.to_datetime(tmp_stamp.date)

        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=_infer_freq_for_pandas(self.freq))
        df_stamp = pd.DataFrame({"date": list(tmp_stamp.date.values) + list(pred_dates[1:])})

        if self.timeenc == 0:
            data_stamp = _make_calendar_features(df_stamp.date.values, freq=self.freq)
        else:
            data_stamp = time_features(df_stamp["date"].values, freq=self.freq).transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = df_data.values[border1:border2] if self.inverse else data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_x[r_begin:r_begin + self.label_len] if self.inverse else self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return (
            torch.tensor(seq_x, dtype=torch.float32),
            torch.tensor(seq_y, dtype=torch.float32),
            torch.tensor(seq_x_mark, dtype=torch.float32),
            torch.tensor(seq_y_mark, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


def infer_data_name(data_path):
    name = os.path.splitext(os.path.basename(data_path))[0].lower()
    if name in ["etth1", "etth2", "ettm1", "ettm2"]:
        return name
    return "custom"


def get_dataset(
    root_path=".",
    flag="train",
    seq_len=96,
    label_len=48,
    pred_len=96,
    features="M",
    data_path="ETTh1.csv",
    target="OT",
    scale=True,
    embed="timeF",
    freq="h",
    data_name=None,
    split_ratios=(0.7, 0.1, 0.2),
    cols: Optional[Sequence[str]] = None,
):
    timeenc = 0 if embed != "timeF" else 1
    data_name = infer_data_name(data_path) if data_name is None else data_name

    return ForecastDataset(
        root_path=root_path,
        flag=flag,
        size=[seq_len, label_len, pred_len],
        features=features,
        data_path=data_path,
        target=target,
        scale=scale,
        timeenc=timeenc,
        freq=freq,
        data_name=data_name,
        split_ratios=split_ratios,
        cols=cols,
    )


def get_data_loaders(
    file_path=None,
    root_path=None,
    data_path=None,
    seq_len=96,
    label_len=48,
    pred_len=96,
    batch_size=32,
    features="M",
    target="OT",
    scale=True,
    embed="timeF",
    freq="h",
    data_name=None,
    split_ratios=(0.7, 0.1, 0.2),
    num_workers=0,
    drop_last=True,
    cols: Optional[Sequence[str]] = None,
):
    """
    Backward-compatible helper.

    Old call still works:
        get_data_loaders(file_path, seq_len, pred_len, batch_size)

    Recommended call:
        get_data_loaders(
            root_path='./dataset/ETT-small',
            data_path='ETTh1.csv',
            seq_len=96,
            label_len=48,
            pred_len=96,
            features='M',
            target='OT',
            embed='timeF',
            freq='h'
        )
    """
    if file_path is not None:
        root_path = os.path.dirname(file_path) or "."
        data_path = os.path.basename(file_path)

    if root_path is None or data_path is None:
        raise ValueError("Either file_path or both root_path and data_path must be provided.")

    common_kwargs = dict(
        root_path=root_path,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        features=features,
        data_path=data_path,
        target=target,
        scale=scale,
        embed=embed,
        freq=freq,
        data_name=data_name,
        split_ratios=split_ratios,
        cols=cols,
    )

    train_dataset = get_dataset(flag="train", **common_kwargs)
    val_dataset = get_dataset(flag="val", **common_kwargs)
    test_dataset = get_dataset(flag="test", **common_kwargs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=drop_last)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=drop_last)

    return train_loader, val_loader, test_loader, train_dataset.scaler


def data_provider(args, flag):
    """
    THUML-style provider.

    Works for Autoformer, iTransformer, PatchTST, vanilla Transformer,
    and your HC/mHC variants because all of them can consume:
        batch_x, batch_y, batch_x_mark, batch_y_mark
    """
    shuffle_flag = flag == "train"
    drop_last = flag == "train"

    data_set = get_dataset(
        root_path=getattr(args, "root_path", "."),
        flag=flag,
        seq_len=args.seq_len,
        label_len=getattr(args, "label_len", args.seq_len // 2),
        pred_len=args.pred_len,
        features=getattr(args, "features", "M"),
        data_path=getattr(args, "data_path"),
        target=getattr(args, "target", "OT"),
        scale=getattr(args, "scale", True),
        embed=getattr(args, "embed", "timeF"),
        freq=getattr(args, "freq", "h"),
        data_name=getattr(args, "data", None),
        split_ratios=getattr(args, "split_ratios", (0.7, 0.1, 0.2)),
    )

    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        num_workers=getattr(args, "num_workers", 0),
        drop_last=drop_last,
    )

    return data_set, data_loader
