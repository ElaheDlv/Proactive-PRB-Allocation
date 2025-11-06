import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
    "time+length": {"columns": ("Time", "Length"), "mode": "event"},
    "time+delta_t+length": {"columns": ("Time", "delta_t", "Length"), "mode": "event"},
    "uniform-length": {"columns": ("Length",), "mode": "uniform"},
}


def create_event_sequences(
    series: np.ndarray,
    window: int,
    target_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build padded event-history windows without resampling."""

    feature_dim = series.shape[1]
    X, y = [], []

    for idx in range(series.shape[0]):
        start = max(0, idx - window)
        history = series[start:idx]
        if history.shape[0] < window:
            pad = np.zeros((window - history.shape[0], feature_dim), dtype=series.dtype)
            history = np.vstack((pad, history))
        X.append(history)
        y.append(series[idx, target_idx])

    return np.stack(X), np.array(y)


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        fc_ratio: float = 0.5,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.layer_norm = nn.LayerNorm(hidden_dim) if layer_norm else None

        head_hidden = max(1, int(hidden_dim * fc_ratio))
        head: List[nn.Module] = [nn.Linear(hidden_dim, head_hidden), nn.ReLU()]
        if dropout > 0:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(head_hidden, 1))
        self.fc = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        out = self.fc(out)
        return out


def resolve_loss(loss_name: str) -> nn.Module:
    loss_name = loss_name.lower()
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name in {"mae", "l1"}:
        return nn.L1Loss()
    if loss_name in {"smoothl1", "huber"}:
        return nn.SmoothL1Loss()
    raise ValueError(f"Unsupported loss function: {loss_name}")


def resolve_optimizer(
    params,
    optimizer_name: str,
    lr: float,
) -> torch.optim.Optimizer:
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    if optimizer_name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    if optimizer_name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def resolve_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    epochs: int,
    factor: float,
    patience: int,
    min_lr: float,
    t_max: int,
    eta_min: float,
) -> torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    scheduler_name = scheduler_name.lower()
    if scheduler_name in {"none", "off"}:
        return None
    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, t_max or epochs),
            eta_min=eta_min,
        )
    raise ValueError(f"Unsupported LR scheduler: {scheduler_name}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    epochs: int,
    device: torch.device,
    lr: float = 1e-3,
    optimizer_name: str = "adam",
    loss_name: str = "mse",
    scheduler_name: str = "none",
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 10,
    scheduler_min_lr: float = 1e-6,
    scheduler_t_max: int = 0,
    scheduler_eta_min: float = 0.0,
    patience: int = 0,
) -> Tuple[List[float], List[float]]:
    criterion = resolve_loss(loss_name)
    optimizer = resolve_optimizer(model.parameters(), optimizer_name=optimizer_name, lr=lr)
    scheduler = resolve_scheduler(
        optimizer,
        scheduler_name=scheduler_name,
        epochs=epochs,
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=scheduler_min_lr,
        t_max=scheduler_t_max,
        eta_min=scheduler_eta_min,
    )
    model.to(device)

    train_history: List[float] = []
    val_history: List[float] = []
    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            preds = model(xb)
            loss = criterion(preds, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= max(1, len(train_loader))
        train_history.append(epoch_loss)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    val_loss += loss.item()
            val_loss /= max(1, len(val_loader))
            val_history.append(val_loss)
            print(
                f"Epoch {epoch + 1}/{epochs} - train_loss={epoch_loss:.6f} val_loss={val_loss:.6f} lr={optimizer.param_groups[0]['lr']:.6g}"
            )

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if patience > 0 and wait >= patience:
                    print("Early stopping triggered.")
                    break

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
        else:
            print(
                f"Epoch {epoch + 1}/{epochs} - train_loss={epoch_loss:.6f} lr={optimizer.param_groups[0]['lr']:.6g}"
            )
            if scheduler is not None and not isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    return train_history, val_history


def plot_loss_curves(
    train_history: List[float],
    val_history: List[float],
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 4))
    epochs_range = range(1, len(train_history) + 1)
    plt.plot(epochs_range, train_history, label="Train")
    if val_history:
        plt.plot(range(1, len(val_history) + 1), val_history, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


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
    train_ratio: float | None,
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Scale selected columns and build padded sequences.

    The scaler is fit on the training portion of the series (chronological)
    when ``train_ratio`` is provided, preventing look-ahead leakage while still
    applying the learned ranges to the entire sequence history.
    """

    scaler = MinMaxScaler()
    values = df[list(feature_cols)].values.astype(np.float32)

    if train_ratio is not None and 0.0 < train_ratio < 1.0:
        train_rows = max(window, int(len(values) * train_ratio))
        train_rows = max(1, min(train_rows, len(values)))
        scaler.fit(values[:train_rows])
        scaled = scaler.transform(values).astype(np.float32)
    else:
        scaled = scaler.fit_transform(values).astype(np.float32)

    target_idx = feature_cols.index("Length")
    X_np, y_np = create_event_sequences(scaled, window=window, target_idx=target_idx)
    return X_np.astype(np.float32), y_np.astype(np.float32), scaler


def inverse_length_transform(
    scaled_values: np.ndarray,
    scaler: MinMaxScaler,
    feature_cols: Sequence[str],
) -> np.ndarray:
    target_idx = feature_cols.index("Length")
    data_min = scaler.data_min_[target_idx]
    data_range = scaler.data_range_[target_idx]
    return scaled_values * data_range + data_min


def predict_in_batches(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu())
            targets.append(yb.cpu())
    pred_cat = torch.cat(preds, dim=0)
    target_cat = torch.cat(targets, dim=0)
    return pred_cat.numpy().squeeze(-1), target_cat.numpy().squeeze(-1)


def format_suffix(**kwargs: object) -> str:
    parts = []
    for key, value in kwargs.items():
        if isinstance(value, float):
            if abs(value) >= 1 and abs(value - round(value)) < 1e-9:
                value_str = str(int(round(value)))
            elif abs(value) < 1e-2:
                value_str = f"{value:.0e}"
            else:
                value_str = f"{value:.4f}".rstrip("0").rstrip(".")
            value_str = value_str.replace("+", "p")
        else:
            value_str = str(value)
        parts.append(f"{key}{value_str}")
    return "_".join(parts)


def train_feature_set(
    df_event: pd.DataFrame,
    trace_path: Path,
    feature_cols: Sequence[str],
    feature_tag: str,
    feature_display: str,
    mode: str,
    window: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    fc_ratio: float,
    layer_norm: bool,
    val_ratio: float,
    patience: int,
    learning_rate: float,
    optimizer_name: str,
    loss_name: str,
    scheduler_name: str,
    scheduler_factor: float,
    scheduler_patience: int,
    scheduler_min_lr: float,
    scheduler_t_max: int,
    scheduler_eta_min: float,
    output_dir: Path,
) -> None:
    if "Length" not in feature_cols:
        raise ValueError("Feature set must include 'Length'.")

    if mode == "uniform":
        df_prepared = build_uniform_dataframe(df_event)
    else:
        df_prepared = df_event

    dropout = float(np.clip(dropout, 0.0, 0.95))
    fc_ratio = max(1e-3, float(fc_ratio))

    val_ratio_clamped = float(np.clip(val_ratio, 0.0, 0.5)) if val_ratio > 0.0 else 0.0
    train_ratio = None if val_ratio_clamped <= 0.0 else 1.0 - val_ratio_clamped

    X_np, y_np, scaler = build_scaled_sequences(
        df_prepared,
        feature_cols=feature_cols,
        window=window,
        train_ratio=train_ratio,
    )

    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.float32).unsqueeze(-1)

    n_samples = X_tensor.size(0)
    split_idx = n_samples
    val_loader = None
    train_loader = None

    if val_ratio_clamped > 0.0:
        split_idx = max(1, int(n_samples * (1 - val_ratio_clamped)))
        val_dataset = TensorDataset(X_tensor[split_idx:], y_tensor[split_idx:])
        if len(val_dataset) > 0:
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_dataset = TensorDataset(X_tensor[:split_idx], y_tensor[:split_idx])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataset = TensorDataset(X_tensor, y_tensor)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMModel(
        input_dim=len(feature_cols),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        fc_ratio=fc_ratio,
        layer_norm=layer_norm,
    )
    train_hist, val_hist = train_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        device=device,
        lr=learning_rate,
        optimizer_name=optimizer_name,
        loss_name=loss_name,
        scheduler_name=scheduler_name,
        scheduler_factor=scheduler_factor,
        scheduler_patience=scheduler_patience,
        scheduler_min_lr=scheduler_min_lr,
        scheduler_t_max=scheduler_t_max,
        scheduler_eta_min=scheduler_eta_min,
        patience=patience,
    )
    common_suffix = format_suffix(
        epochs=epochs,
        window=window,
        batch=batch_size,
        hidden=hidden_dim,
        layers=num_layers,
        drop=dropout,
        fcr=fc_ratio,
        ln=int(layer_norm),
        lr=learning_rate,
        opt=optimizer_name,
        loss=loss_name,
        sched=scheduler_name,
    )
    loss_plot = output_dir / f"{trace_path.stem}_{feature_tag}_loss_{common_suffix}.png"
    loss_title = (
        f"Loss - {trace_path.stem} [{feature_display}]\n"
        f"opt={optimizer_name} lr={learning_rate} loss={loss_name} sched={scheduler_name} "
        f"layers={num_layers} hidden={hidden_dim} drop={dropout} fcr={fc_ratio:.2f} ln={int(layer_norm)}"
    )
    plot_loss_curves(train_hist, val_hist, loss_title, loss_plot)

    preds, y_true = predict_in_batches(model, eval_loader, device=device)

    preds = inverse_length_transform(preds, scaler, feature_cols)
    y_true = inverse_length_transform(y_true, scaler, feature_cols)

    time_axis = df_prepared["Time"].values.astype(float)
    title = (
        f"{feature_display} [{mode}] - {trace_path.stem} - {epochs} epochs\n"
        f"opt={optimizer_name} lr={learning_rate} loss={loss_name} sched={scheduler_name} "
        f"layers={num_layers} hidden={hidden_dim} drop={dropout} fcr={fc_ratio:.2f} ln={int(layer_norm)}"
    )
    output_path = output_dir / f"{trace_path.stem}_{feature_tag}_pred_{common_suffix}.png"
    plot_predictions(time_axis, y_true, preds, title, output_path, is_irregular="delta_t" in feature_cols)


def load_trace(trace_path: Path) -> pd.DataFrame:
    df = pd.read_csv(trace_path)
    df = df.sort_values("Time").reset_index(drop=True)
    if "delta_t" not in df.columns:
        df["delta_t"] = df["Time"].diff().fillna(0.0)
    return df


def build_uniform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Expand the event stream onto a uniform grid based on the smallest Δt."""

    if df.empty:
        return df.copy()

    time_vals = df["Time"].astype(float).to_numpy()
    deltas = np.diff(time_vals)
    positive_deltas = deltas[deltas > 0]
    if positive_deltas.size == 0:
        min_gap = 1.0
    else:
        min_gap = positive_deltas.min()

    if min_gap <= 0:
        min_gap = 1.0

    start = float(time_vals[0])
    stop = float(time_vals[-1])
    total_span = max(stop - start, min_gap)
    max_steps = 2_000_000

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
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension of the LSTM.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate applied within LSTM stack and head.")
    parser.add_argument(
        "--fc-ratio",
        type=float,
        default=0.5,
        help="Hidden fraction for the MLP head (multiplier * hidden_dim).",
    )
    parser.add_argument(
        "--no-layer-norm",
        action="store_false",
        dest="layer_norm",
        default=True,
        help="Disable layer normalization before the prediction head.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of samples used for validation (chronological split).")
    parser.add_argument("--early-stop", type=int, default=0, help="Early stopping patience based on validation loss (0 disables).")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=("adam", "adamw", "rmsprop", "sgd"),
        help="Optimizer to use (default: adam).",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        choices=("mse", "mae", "l1", "smoothl1", "huber"),
        help="Regression loss to minimize (default: mse).",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="none",
        choices=("none", "plateau", "cosine"),
        help="Learning rate scheduler to apply (default: none).",
    )
    parser.add_argument(
        "--lr-factor",
        type=float,
        default=0.5,
        help="Multiplicative factor for ReduceLROnPlateau (used when --lr-scheduler=plateau).",
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=10,
        help="Patience (epochs) before LR is reduced (plateau scheduler).",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=1e-6,
        help="Minimum LR allowed by the scheduler (plateau scheduler).",
    )
    parser.add_argument(
        "--lr-tmax",
        type=int,
        default=0,
        help="T_max parameter for cosine annealing (defaults to epochs when 0).",
    )
    parser.add_argument(
        "--lr-eta-min",
        type=float,
        default=0.0,
        help="Minimum LR for cosine annealing scheduler.",
    )
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
        feature_display = " + ".join(cols)
        feature_tag = feature_name.replace(" ", "")
        print(
            " ".join(
                (
                    f"Training feature set '{feature_name}' ({mode}) -> {feature_display}",
                    f"optimizer={args.optimizer}",
                    f"lr={args.learning_rate}",
                    f"loss={args.loss}",
                    f"scheduler={args.lr_scheduler}",
                    f"layers={args.num_layers}",
                    f"hidden={args.hidden_dim}",
                    f"dropout={args.dropout}",
                    f"layer_norm={args.layer_norm}",
                )
            )
        )
        train_feature_set(
            df_event=df_event,
            trace_path=args.trace,
            feature_cols=cols,
            feature_tag=feature_tag,
            feature_display=feature_display,
            mode=mode,
            window=args.window,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            fc_ratio=args.fc_ratio,
            layer_norm=args.layer_norm,
            val_ratio=args.val_ratio,
            patience=args.early_stop,
            learning_rate=args.learning_rate,
            optimizer_name=args.optimizer,
            loss_name=args.loss,
            scheduler_name=args.lr_scheduler,
            scheduler_factor=args.lr_factor,
            scheduler_patience=args.lr_patience,
            scheduler_min_lr=args.lr_min,
            scheduler_t_max=args.lr_tmax,
            scheduler_eta_min=args.lr_eta_min,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
