import argparse
import copy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


FEATURE_SETS: Dict[str, Dict[str, Sequence[str]]] = {
    "length": {"columns": ("Length",), "mode": "event"},
    "delta_t+length": {"columns": ("delta_t", "Length"), "mode": "event"},
    "uniform-length": {"columns": ("Length",), "mode": "uniform"},
}


def create_event_sequences(
    series: np.ndarray,
    window: int,
    target_idx: int,
    target_array: Optional[np.ndarray] = None,
    include_mask: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build padded event-history windows without resampling."""

    feature_dim = series.shape[1]
    out_dim = feature_dim + (1 if include_mask else 0)
    X, y = [], []

    for idx in range(series.shape[0]):
        start = max(0, idx - window)
        actual_len = idx - start
        history = series[start:idx]
        if actual_len < window:
            pad = np.zeros((window - actual_len, feature_dim), dtype=series.dtype)
            history = np.vstack((pad, history))
        mask_col = None
        if include_mask:
            mask_col = np.zeros((window, 1), dtype=series.dtype)
            if actual_len > 0:
                mask_col[-actual_len:] = 1.0
            history = np.concatenate((history, mask_col), axis=1)
        X.append(history)
        if target_array is not None:
            y.append(target_array[idx])
        else:
            y.append(series[idx, target_idx])

    return np.stack(X), np.array(y)


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        fc_hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_dim
        if fc_hidden:
            self.head = nn.Sequential(
                nn.Linear(lstm_out_dim, fc_hidden),
                nn.ReLU(),
                nn.Linear(fc_hidden, 1),
            )
        else:
            self.head = nn.Linear(lstm_out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


def train_model(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    device: torch.device,
    lr: float = 1e-3,
) -> None:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            preds = model(xb)
            loss = criterion(preds, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= max(1, len(loader))
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.6f}")


def plot_predictions(
    time_axis: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    output_path: Path,
    is_irregular: bool,
) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(time_axis, y_true, label="True")
    plt.plot(time_axis, y_pred, label="Predicted")
    plt.title(title)
    plt.xlabel("Time [ms]" if not is_irregular else "Time [ms] (irregular sampling)")
    plt.ylabel("Length [bytes]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot -> {output_path}")


def build_scaled_sequences(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    window: int,
    include_mask: bool,
    target_mode: str,
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, int]:
    """Scale selected columns and build padded sequences."""

    scaler = MinMaxScaler()
    values = df[list(feature_cols)].values.astype(np.float32)
    scaled = scaler.fit_transform(values).astype(np.float32)

    target_idx = feature_cols.index("Length")

    if target_mode == "scaled":
        target_vals = scaled[:, target_idx]
    else:
        target_vals = df["Length"].values.astype(np.float32)
        if target_mode == "log":
            target_vals = np.log1p(np.maximum(target_vals, 0.0))

    X_np, y_np = create_event_sequences(
        scaled,
        window=window,
        target_idx=target_idx,
        target_array=target_vals,
        include_mask=include_mask,
    )
    return X_np.astype(np.float32), y_np.astype(np.float32), scaler, target_idx


def inverse_target_transform(
    values: np.ndarray,
    mode: str,
    scaler: MinMaxScaler,
    target_idx: int,
) -> np.ndarray:
    if mode == "log":
        return np.expm1(values)
    if mode == "scaled":
        data_min = scaler.data_min_[target_idx]
        data_range = scaler.data_range_[target_idx]
        return values * data_range + data_min
    return values


def predict_in_batches(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    return_targets: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if return_targets:
                xb, yb = batch[:2]
                targets.append(yb.cpu())
            else:
                xb = batch[0]
            xb = xb.to(device)
            preds.append(model(xb).cpu())
    pred_cat = torch.cat(preds, dim=0).numpy().squeeze(-1)
    target_cat = None
    if return_targets and targets:
        target_cat = torch.cat(targets, dim=0).numpy().squeeze(-1)
    return pred_cat, target_cat


def train_feature_set(
    df_event: pd.DataFrame,
    trace_path: Path,
    feature_cols: Sequence[str],
    feature_tag: str,
    mode: str,
    window: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    fc_hidden: int | None,
    include_pad_mask: bool,
    target_mode: str,
    loss_type: str,
    huber_delta: float,
    zero_weight: float,
    clip_grad: float,
    val_ratio: float,
    early_stop: int,
    plateau_patience: int,
    output_dir: Path,
) -> None:
    if "Length" not in feature_cols:
        raise ValueError("Feature set must include 'Length'.")

    if mode == "uniform":
        df_prepared = build_uniform_dataframe(df_event)
        mask_flag = False
    else:
        df_prepared = df_event
        mask_flag = include_pad_mask

    X_np, y_proc, scaler, target_idx = build_scaled_sequences(
        df_prepared,
        feature_cols=feature_cols,
        window=window,
        include_mask=mask_flag,
        target_mode=target_mode,
    )

    if target_mode == "scaled":
        target_raw = df_prepared["Length"].values.astype(np.float32)
    elif target_mode == "log":
        target_raw = np.log1p(np.maximum(df_prepared["Length"].values.astype(np.float32), 0.0))
    else:
        target_raw = df_prepared["Length"].values.astype(np.float32)
    y_np = y_proc

    weights_np: Optional[np.ndarray] = None
    if zero_weight < 1.0:
        weights_np = np.ones_like(y_np, dtype=np.float32)
        mask_zero = target_raw <= 0.0
        weights_np[mask_zero] = zero_weight

    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.float32).unsqueeze(-1)

    n_samples = X_np.shape[0]
    split_idx = n_samples
    val_ratio = float(np.clip(val_ratio, 0.0, 0.5))
    if val_ratio > 0.0:
        split_idx = max(1, int(n_samples * (1 - val_ratio)))
    train_slice = slice(0, split_idx)
    val_slice = slice(split_idx, None)

    train_tensors = [X_tensor[train_slice], y_tensor[train_slice]]
    if weights_np is not None:
        weight_tensor = torch.tensor(weights_np[train_slice], dtype=torch.float32)
        train_tensors.append(weight_tensor)
    train_dataset = TensorDataset(*train_tensors)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = None
    if split_idx < n_samples:
        val_dataset = TensorDataset(
            X_tensor[val_slice],
            y_tensor[val_slice],
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    full_dataset = TensorDataset(X_tensor, y_tensor)
    eval_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMModel(
        input_dim=X_np.shape[-1],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        fc_hidden=fc_hidden,
    )

    criterion = None
    loss_type = loss_type.lower()
    if loss_type == "mse":
        criterion = nn.MSELoss(reduction="none")
    elif loss_type == "mae":
        criterion = nn.L1Loss(reduction="none")
    elif loss_type == "huber":
        criterion = nn.SmoothL1Loss(beta=huber_delta, reduction="none")
    else:
        raise ValueError(f"Unsupported loss: {loss_type}")

    optimizer = torch.optim.Adam(model.parameters())
    scheduler = None
    if plateau_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=int(plateau_patience),
            factor=0.5,
        )

    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    model.to(device)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for batch in train_loader:
            if weights_np is not None:
                xb, yb, wb = batch
                wb = wb.to(device)
            else:
                xb, yb = batch
                wb = None
            xb = xb.to(device)
            yb = yb.to(device)

            preds = model(xb)
            loss_vec = criterion(preds, yb)
            loss_vec = loss_vec.view(loss_vec.size(0), -1)
            if wb is not None:
                loss = (loss_vec.mean(dim=1) * wb).mean()
            else:
                loss = loss_vec.mean()

            optimizer.zero_grad()
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        epoch_loss /= max(1, batches)

        val_loss = None
        if val_loader is not None:
            model.eval()
            cumulative = 0.0
            vbatches = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    preds = model(xb)
                    loss_vec = criterion(preds, yb)
                    loss = loss_vec.mean()
                    cumulative += loss.item()
                    vbatches += 1
            val_loss = cumulative / max(1, vbatches)

        monitored = val_loss if val_loss is not None else epoch_loss
        if scheduler is not None:
            scheduler.step(monitored)

        print(
            f"Epoch {epoch + 1}/{epochs} - train_loss={epoch_loss:.6f}"
            + (f" val_loss={val_loss:.6f}" if val_loss is not None else "")
        )

        if monitored < best_loss - 1e-6:
            best_loss = monitored
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if early_stop > 0 and patience_counter >= early_stop:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    preds, y_true_transformed = predict_in_batches(model, eval_loader, device=device)

    preds = inverse_target_transform(preds, target_mode, scaler, target_idx)
    if y_true_transformed is not None:
        y_true = inverse_target_transform(y_true_transformed, target_mode, scaler, target_idx)
    else:
        y_true = np.zeros_like(preds)

    time_axis = df_prepared["Time"].values.astype(float)
    title = f"Seq ({' + '.join(feature_cols)}) [{mode}] - {trace_path.stem} - {epochs} epochs"
    output_path = output_dir / f"{trace_path.stem}_epochs{epochs}_{feature_tag}.png"
    plot_predictions(time_axis, y_true, preds, title, output_path, is_irregular="delta_t" in feature_cols)


def load_trace(trace_path: Path) -> pd.DataFrame:
    df = pd.read_csv(trace_path)
    df = df.sort_values("Time").reset_index(drop=True)
    if "delta_t" not in df.columns:
        df["delta_t"] = df["Time"].diff().fillna(0.0)
    return df


def build_uniform_dataframe(
    df: pd.DataFrame,
    quantile: float = 0.05,
    max_steps: int = 2_000_000,
) -> pd.DataFrame:
    """Expand the event stream onto a uniform grid based on the smallest Δt."""

    if df.empty:
        return df.copy()

    time_vals = df["Time"].astype(float).to_numpy()
    deltas = np.diff(time_vals)
    positive_deltas = deltas[deltas > 0]
    if positive_deltas.size == 0:
        min_gap = 1.0
    else:
        q = np.clip(quantile, 0.0, 1.0)
        min_gap = float(np.quantile(positive_deltas, q)) if q > 0 else float(positive_deltas.min())

    if min_gap <= 0:
        min_gap = 1.0

    start = float(time_vals[0])
    stop = float(time_vals[-1])
    total_span = max(stop - start, min_gap)

    steps = int(np.floor(total_span / min_gap)) + 1
    if steps > max_steps:
        min_gap = total_span / max_steps
        steps = max_steps + 1

    grid = start + np.arange(steps) * min_gap
    length_series = np.zeros_like(grid)

    for t, length in zip(time_vals, df["Length"].to_numpy(dtype=float)):
        idx = int(round((t - start) / min_gap))
        if 0 <= idx < steps:
            length_series[idx] += length

    uniform_df = pd.DataFrame({"Time": grid, "Length": length_series})
    uniform_df["delta_t"] = min_gap
    return uniform_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM forecasters on telecom traces.")
    parser.add_argument("trace", type=Path, help="Path to the CSV trace file.")
    parser.add_argument("--window", type=int, default=20, help="Sliding window size.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs for each model.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension of the LSTM.")
    parser.add_argument("--num-layers", type=int, default=1, help="Number of LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout between LSTM layers (applies when num_layers>1).")
    parser.add_argument("--fc-hidden", type=int, default=None, help="Optional hidden units for an extra fully-connected layer after the LSTM.")
    parser.add_argument("--include-pad-mask", action="store_true", help="Append a padding-mask channel for event-mode sequences (helps the model distinguish real zeros from left-padding).")
    parser.add_argument("--target-mode", type=str, default="raw", choices=("raw", "scaled", "log"), help="How to transform the prediction target: raw bytes, min-max scaled, or log1p.")
    parser.add_argument("--loss", type=str, default="mse", choices=("mse", "mae", "huber"), help="Regression loss to optimize.")
    parser.add_argument("--huber-delta", type=float, default=1.0, help="Delta parameter for Huber loss.")
    parser.add_argument("--zero-weight", type=float, default=1.0, help="Relative weight for zero-length targets (helps with class imbalance).")
    parser.add_argument("--clip-grad", type=float, default=0.0, help="Gradient clipping value (0 disables).")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of samples reserved for validation (chronological split).")
    parser.add_argument("--early-stop", type=int, default=0, help="Early stopping patience based on validation loss (0 disables).")
    parser.add_argument("--plateau-patience", type=int, default=0, help="Patience for ReduceLROnPlateau scheduler (0 disables).")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Torch device to use. Default picks CUDA if available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory to store generated plots.",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=["length", "delta_t+length"],
        choices=sorted(FEATURE_SETS.keys()),
        help="One or more feature sets to train (default: length and delta_t+length).",
    )
    return parser.parse_args()


def resolve_device(arg: str) -> torch.device:
    if arg == "cuda":
        return torch.device("cuda")
    if arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df_event = load_trace(args.trace)

    for feature_name in args.feature_sets:
        config = FEATURE_SETS[feature_name]
        cols = config["columns"]
        mode = config.get("mode", "event")
        print(f"Training feature set '{feature_name}' ({mode}) -> {cols}")
        train_feature_set(
            df_event=df_event,
            trace_path=args.trace,
            feature_cols=cols,
            feature_tag=feature_name.replace("+", "_").replace(" ", ""),
            mode=mode,
            window=args.window,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            fc_hidden=args.fc_hidden,
            include_pad_mask=args.include_pad_mask,
            target_mode=args.target_mode,
            loss_type=args.loss,
            huber_delta=args.huber_delta,
            zero_weight=args.zero_weight,
            clip_grad=args.clip_grad,
            val_ratio=args.val_ratio,
            early_stop=args.early_stop,
            plateau_patience=args.plateau_patience,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
