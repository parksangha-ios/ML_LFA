from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


CLASS_NAMES = {0: "Parker", 1: "Cowan-like", 2: "Cape-like"}


def plot_history(history_path: str, out_dir: Path) -> None:
    hist = json.loads(Path(history_path).read_text(encoding="utf-8"))
    df = pd.DataFrame(hist)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(df["epoch"], df["train_loss"], label="train_loss")
    axes[0].plot(df["epoch"], df["val_loss"], label="val_loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(df["epoch"], df["train_cls_acc"], label="train_cls_acc")
    axes[1].plot(df["epoch"], df["val_cls_acc"], label="val_cls_acc")
    axes[1].set_title("Classification Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(df["epoch"], df["val_alpha_mape"], color="tab:red")
    axes[2].set_title("Validation Alpha MAPE")
    axes[2].set_xlabel("Epoch")

    fig.tight_layout()
    fig.savefig(out_dir / "goal2_history_curves.png", dpi=180)
    plt.close(fig)


def plot_diagnostic(report_path: str, pred_path: str, out_dir: Path) -> None:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    pred = pd.read_csv(pred_path)

    # confusion matrix heatmap
    cm = np.array(report["confusion_matrix"], dtype=int)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=[CLASS_NAMES[i] for i in range(3)],
        yticklabels=[CLASS_NAMES[i] for i in range(3)],
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out_dir / "goal2_confusion_matrix.png", dpi=180)
    plt.close(fig)

    # alpha error histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(pred["alpha_abs_rel_err"], bins=50, kde=True, ax=ax)
    ax.axvline(float(report["alpha_mape_mean"]), color="r", linestyle="--", label="mean MAPE")
    ax.axvline(float(report["alpha_mape_median"]), color="g", linestyle="--", label="median MAPE")
    ax.set_title("Alpha Absolute Relative Error Distribution")
    ax.set_xlabel("|alpha_pred - alpha_true| / alpha_true")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "goal2_alpha_error_hist.png", dpi=180)
    plt.close(fig)

    # true vs pred scatter (log scale)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for cls in [0, 1, 2]:
        d = pred[pred["y_true"] == cls]
        ax.scatter(d["alpha_true"], d["alpha_pred"], s=8, alpha=0.35, label=CLASS_NAMES[cls])

    vmin = float(min(pred["alpha_true"].min(), pred["alpha_pred"].min()))
    vmax = float(max(pred["alpha_true"].max(), pred["alpha_pred"].max()))
    ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("True alpha [mm^2/s]")
    ax.set_ylabel("Predicted alpha [mm^2/s]")
    ax.set_title("Alpha: True vs Predicted")
    ax.legend(markerscale=2)
    fig.tight_layout()
    fig.savefig(out_dir / "goal2_alpha_scatter.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Goal2 training and diagnostics")
    parser.add_argument("--history", type=str, default="history_mtl_heatloss.json")
    parser.add_argument("--report", type=str, default="goal2_diagnostic_report.json")
    parser.add_argument("--pred", type=str, default="goal2_predictions.csv")
    parser.add_argument("--out-dir", type=str, default="artifacts")
    args, _unknown = parser.parse_known_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_history(args.history, out_dir)
    plot_diagnostic(args.report, args.pred, out_dir)
    print(f"saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
