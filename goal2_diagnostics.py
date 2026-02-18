from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, random_split

from mtl_heatloss_pytorch import LFAMTLDataset, LFAMTLNet, SEED, SynthConfig, make_dataset


@dataclass
class DiagnosticSummary:
    class_counts: dict
    class_balance_ratio: float
    alpha_hist_cv: float
    l_hist_cv: float
    sample_feature_ratio: float
    pca_dim_95: int
    effective_rank: float
    val_acc: float
    val_macro_f1: float
    alpha_mape_mean: float
    alpha_mape_median: float
    alpha_mape_p90: float


def _coefficient_of_variation(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.std(x) / (np.mean(x) + 1e-12))


def data_balance_and_dimensionality(data: dict[str, np.ndarray], n_bins: int = 12):
    cls = data["curve_type"]
    uniq, cnt = np.unique(cls, return_counts=True)
    class_counts = {int(k): int(v) for k, v in zip(uniq, cnt)}
    balance_ratio = float(np.min(cnt) / np.max(cnt))

    alpha_hist, _ = np.histogram(data["alpha_log"].ravel(), bins=n_bins)
    l_hist, _ = np.histogram(data["l_log"].ravel(), bins=n_bins)
    alpha_cv = _coefficient_of_variation(alpha_hist)
    l_cv = _coefficient_of_variation(l_hist)

    features = np.concatenate([data["v"], data["t_log"], data["l_log"]], axis=1)
    n_samples, n_features = features.shape
    sample_feature_ratio = n_samples / n_features

    z = StandardScaler().fit_transform(features)
    pca = PCA().fit(z)
    csum = np.cumsum(pca.explained_variance_ratio_)
    pca_dim_95 = int(np.searchsorted(csum, 0.95) + 1)

    singular_vals = pca.singular_values_
    p = singular_vals / (singular_vals.sum() + 1e-12)
    effective_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))

    return {
        "class_counts": class_counts,
        "class_balance_ratio": balance_ratio,
        "alpha_hist_cv": alpha_cv,
        "l_hist_cv": l_cv,
        "sample_feature_ratio": sample_feature_ratio,
        "pca_dim_95": pca_dim_95,
        "effective_rank": effective_rank,
    }


def evaluate_model(data: dict[str, np.ndarray], ckpt: str, n_points: int, train_ratio: float = 0.8):
    ds = LFAMTLDataset(data)
    n_train = int(len(ds) * train_ratio)
    train_ds, val_ds = random_split(ds, [n_train, len(ds) - n_train], generator=torch.Generator().manual_seed(SEED))

    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LFAMTLNet(n_points=n_points).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    ys, ps = [], []
    alpha_true_log, alpha_pred_log = [], []

    with torch.no_grad():
        for batch in val_loader:
            out = model(batch["v"].to(device), batch["t_log"].to(device), batch["l_log"].to(device))
            pred_cls = out["curve_logits"].argmax(dim=1).cpu().numpy()

            ys.append(batch["curve_type"].cpu().numpy())
            ps.append(pred_cls)

            alpha_true_log.append(batch["alpha_log"].cpu().numpy())
            alpha_pred_log.append(out["alpha_log"].cpu().numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    t_log = np.concatenate(alpha_true_log, axis=0)
    p_log = np.concatenate(alpha_pred_log, axis=0)
    t = 10 ** t_log
    p = 10 ** p_log
    mape = np.abs(p - t) / np.maximum(t, 1e-12)

    return {
        "val_acc": float(accuracy_score(y_true, y_pred)),
        "val_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": cm.tolist(),
        "alpha_mape_mean": float(np.mean(mape)),
        "alpha_mape_median": float(np.median(mape)),
        "alpha_mape_p90": float(np.quantile(mape, 0.9)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Goal2 model diagnostics: balance + dimensionality + metrics")
    parser.add_argument("--n-samples", type=int, default=6000)
    parser.add_argument("--n-points", type=int, default=128)
    parser.add_argument("--checkpoint", type=str, default="best_mtl_heatloss.pt")
    parser.add_argument("--output", type=str, default="goal2_diagnostic_report.json")
    args, _unknown = parser.parse_known_args()

    cfg = SynthConfig(n_samples=args.n_samples, n_points=args.n_points)
    data = make_dataset(cfg)

    report = {}
    report.update(data_balance_and_dimensionality(data))
    report.update(evaluate_model(data, args.checkpoint, n_points=args.n_points))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"saved report: {args.output}")


if __name__ == "__main__":
    main()
