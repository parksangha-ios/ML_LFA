from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


SEED = 42
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)


@dataclass
class SynthConfig:
    n_samples: int = 24000
    n_points: int = 128
    alpha_min: float = 0.01      # mm^2/s
    alpha_max: float = 2000.0    # mm^2/s
    l_min: float = 0.2           # mm
    l_max: float = 6.0           # mm
    tau_max: float = 1.3


def parker_theta(tau: np.ndarray, n_terms: int = 120) -> np.ndarray:
    m = np.arange(n_terms, dtype=np.float64)[:, None]
    n = 2.0 * m + 1.0
    expo = np.exp(-(n**2) * (math.pi**2) * tau[None, :])
    theta = 1.0 - (8.0 / (math.pi**2)) * (expo / (n**2)).sum(axis=0)
    return np.clip(theta, 0.0, 1.0)


def finite_pulse_convolution(theta: np.ndarray, pulse_steps: int) -> np.ndarray:
    if pulse_steps <= 1:
        return theta
    kernel = np.ones(pulse_steps, dtype=np.float64) / pulse_steps
    out = np.convolve(theta, kernel, mode="same")
    return np.clip(out, 0.0, 1.0)


def build_single_curve(curve_type: int, tau: np.ndarray) -> tuple[np.ndarray, float]:
    """
    curve_type: 0=Parker, 1=Cowan-like(loss), 2=Cape-Lehman-like(finite pulse + loss)
    """
    base = parker_theta(tau)

    if curve_type == 0:
        loss_strength = 0.0
        out = base
    elif curve_type == 1:
        loss_strength = float(rng.uniform(0.12, 1.5))
        out = base * np.exp(-loss_strength * tau)
    else:
        pulse_steps = int(rng.integers(3, 14))
        loss_strength = float(rng.uniform(0.05, 1.0))
        pulse_blur = finite_pulse_convolution(base, pulse_steps=pulse_steps)
        out = pulse_blur * np.exp(-loss_strength * tau)

    noise = rng.normal(0.0, 0.004, size=tau.shape[0])
    out = np.clip(out + noise, 0.0, None)

    vmax = float(np.max(out))
    out = out / vmax if vmax > 0 else out
    return out.astype(np.float32), loss_strength


def make_dataset(cfg: SynthConfig) -> dict[str, np.ndarray]:
    tau = np.linspace(0.0, cfg.tau_max, cfg.n_points, dtype=np.float64)

    xs_v = np.zeros((cfg.n_samples, cfg.n_points), dtype=np.float32)
    xs_tlog = np.zeros((cfg.n_samples, cfg.n_points), dtype=np.float32)
    xs_l = np.zeros((cfg.n_samples, 1), dtype=np.float32)

    y_cls = np.zeros(cfg.n_samples, dtype=np.int64)
    y_alpha_log = np.zeros((cfg.n_samples, 1), dtype=np.float32)
    y_loss = np.zeros((cfg.n_samples, 1), dtype=np.float32)

    # 클래스 균형 샘플링: 0/1/2를 순환 배치
    curve_types = np.arange(cfg.n_samples, dtype=np.int64) % 3
    rng.shuffle(curve_types)

    for i in range(cfg.n_samples):
        curve_type = int(curve_types[i])

        alpha_log = float(rng.uniform(math.log10(cfg.alpha_min), math.log10(cfg.alpha_max)))
        alpha = 10.0 ** alpha_log
        l_mm = float(rng.uniform(cfg.l_min, cfg.l_max))

        v, loss_strength = build_single_curve(curve_type=curve_type, tau=tau)
        t = tau * (l_mm**2) / alpha
        t = np.maximum(t, 1e-12)
        t_log = np.log10(t)

        xs_v[i] = v
        xs_tlog[i] = t_log.astype(np.float32)
        xs_l[i, 0] = math.log10(l_mm)
        y_cls[i] = curve_type
        y_alpha_log[i, 0] = alpha_log
        y_loss[i, 0] = loss_strength

    return {
        "v": xs_v,
        "t_log": xs_tlog,
        "l_log": xs_l,
        "curve_type": y_cls,
        "alpha_log": y_alpha_log,
        "loss_strength": y_loss,
    }


class LFAMTLDataset(Dataset):
    def __init__(self, data: dict[str, np.ndarray]) -> None:
        self.v = torch.from_numpy(data["v"])
        self.t_log = torch.from_numpy(data["t_log"])
        self.l_log = torch.from_numpy(data["l_log"])
        self.curve_type = torch.from_numpy(data["curve_type"])
        self.alpha_log = torch.from_numpy(data["alpha_log"])
        self.loss_strength = torch.from_numpy(data["loss_strength"])

    def __len__(self) -> int:
        return self.v.shape[0]

    def __getitem__(self, idx: int):
        return {
            "v": self.v[idx],
            "t_log": self.t_log[idx],
            "l_log": self.l_log[idx],
            "curve_type": self.curve_type[idx],
            "alpha_log": self.alpha_log[idx],
            "loss_strength": self.loss_strength[idx],
        }


class LFAMTLNet(nn.Module):
    def __init__(self, n_points: int, hidden: int = 256) -> None:
        super().__init__()
        in_dim = (2 * n_points) + 1
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(0.1),
        )
        self.class_head = nn.Linear(hidden, 3)
        self.alpha_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        self.loss_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

    def forward(self, v: torch.Tensor, t_log: torch.Tensor, l_log: torch.Tensor):
        x = torch.cat([v, t_log, l_log], dim=1)
        h = self.backbone(x)
        return {
            "curve_logits": self.class_head(h),
            "alpha_log": self.alpha_head(h),
            "loss_strength": self.loss_head(h),
        }


def run_epoch(model, loader, optimizer, device):
    train = optimizer is not None
    model.train(train)

    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    total = 0
    cls_correct = 0
    loss_sum = 0.0

    for batch in loader:
        v = batch["v"].to(device)
        t_log = batch["t_log"].to(device)
        l_log = batch["l_log"].to(device)

        y_cls = batch["curve_type"].to(device)
        y_alpha = batch["alpha_log"].to(device)
        y_loss = batch["loss_strength"].to(device)

        out = model(v, t_log, l_log)

        loss_cls = ce(out["curve_logits"], y_cls)
        loss_alpha = mse(out["alpha_log"], y_alpha)
        mask_non_parker = (y_cls != 0).float().unsqueeze(1)
        loss_loss = ((out["loss_strength"] - y_loss) ** 2 * mask_non_parker).sum() / (mask_non_parker.sum() + 1e-6)

        loss = 1.0 * loss_cls + 1.8 * loss_alpha + 0.6 * loss_loss

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        cls_correct += (out["curve_logits"].argmax(dim=1) == y_cls).sum().item()
        total += y_cls.numel()
        loss_sum += float(loss.item()) * y_cls.shape[0]

    return {"loss": loss_sum / total, "cls_acc": cls_correct / total}


def evaluate_regression(model, loader, device):
    model.eval()
    abs_rel_alpha = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["v"].to(device), batch["t_log"].to(device), batch["l_log"].to(device))
            pred = 10 ** out["alpha_log"].cpu().numpy()
            true = 10 ** batch["alpha_log"].cpu().numpy()
            abs_rel_alpha.append(np.abs(pred - true) / np.maximum(true, 1e-12))
    return float(np.mean(np.concatenate(abs_rel_alpha, axis=0)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MTL model for LFA heat-loss aware alpha prediction")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--n-samples", type=int, default=24000)
    parser.add_argument("--n-points", type=int, default=128)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--save-history", type=str, default="history_mtl_heatloss.json")

    # Jupyter/IPython에서도 안전하게 동작
    args, _unknown = parser.parse_known_args()

    cfg = SynthConfig(n_samples=args.n_samples, n_points=args.n_points)
    data = make_dataset(cfg)
    ds = LFAMTLDataset(data)

    n_train = int(len(ds) * args.train_ratio)
    train_ds, val_ds = random_split(ds, [n_train, len(ds) - n_train], generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LFAMTLNet(cfg.n_points).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    best_val, best_path = float("inf"), "best_mtl_heatloss.pt"
    history = []

    print(f"Starting training on {device}...")
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, optim, device)
        va = run_epoch(model, val_loader, None, device)
        mape = evaluate_regression(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_cls_acc": tr["cls_acc"],
            "val_loss": va["loss"],
            "val_cls_acc": va["cls_acc"],
            "val_alpha_mape": mape,
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d} | Train Loss: {tr['loss']:.4f} | "
            f"Val Acc: {va['cls_acc']:.3f} | Alpha MAPE: {mape:.3f}",
            flush=True,
        )

        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(model.state_dict(), best_path)

    Path(args.save_history).write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Training complete. Best model saved to: {best_path}")
    print(f"History saved to: {args.save_history}")


if __name__ == "__main__":
    main()
