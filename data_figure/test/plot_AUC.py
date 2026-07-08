import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 1.3
plt.rcParams["font.size"] = 14
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.edgecolor"] = "black"

colors = [
    "#fa7f6f",
    "#ffbe7a",
    "#f0d879",
    "#cde58b",
    "#9bbf8a",
    "#b8e3d7",
    "#8dcec8",
    "#add3e2",
    "#82afda",
    "#a7b8ee",
    "#c2bdde",
    "#e8a9d0",
    "#f2a7ad",
]


def load_scores(txt_path):
    scores = []
    if not os.path.exists(txt_path):
        return scores
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 2:
                continue
            s = parts[1].strip()
            if re.match(r"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$", s):
                try:
                    scores.append(float(s))
                except ValueError:
                    pass
    return scores


def get_method_threshold(method, default_threshold, pscore_threshold):
    lower_name = method.lower()
    if "plaac" in lower_name or "llphyscore" in lower_name:
        return None
    if "pscore" in lower_name:
        return pscore_threshold
    return default_threshold


def calculate_metrics(y_true, y_score, threshold):
    auc = roc_auc_score(y_true, y_score)
    if threshold is None:
        return {
            "AUC": auc,
            "ACC": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1": np.nan,
            "MCC": np.nan,
        }
    y_pred = (y_score >= threshold).astype(int)
    return {
        "AUC": auc,
        "ACC": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def collect_methods_from_subdirs(default_threshold, pscore_threshold):
    methods = []
    for name in sorted(os.listdir(".")):
        if not os.path.isdir(name) or name.startswith("."):
            continue

        pos_path = os.path.join(name, "LLPS_score.txt")
        neg_path = os.path.join(name, "non_LLPS_score.txt")
        if not (os.path.exists(pos_path) and os.path.exists(neg_path)):
            continue

        pos_scores = load_scores(pos_path)
        neg_scores = load_scores(neg_path)

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            print(f"Skip {name}: pos={len(pos_scores)}, neg={len(neg_scores)}")
            continue

        y_true = np.array([1] * len(pos_scores) + [0] * len(neg_scores), dtype=int)
        y_score = np.array(pos_scores + neg_scores, dtype=float)
        threshold = get_method_threshold(name, default_threshold, pscore_threshold)

        try:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            metrics = calculate_metrics(y_true, y_score, threshold)
        except Exception as e:
            print(f"Failed to calculate metrics for {name}: {e}")
            continue

        record = {
            "Method": name,
            "fpr": fpr,
            "tpr": tpr,
            "n_pos": len(pos_scores),
            "n_neg": len(neg_scores),
            "threshold": threshold,
        }
        record.update(metrics)
        methods.append(record)

    methods.sort(key=lambda x: x["AUC"], reverse=True)
    return methods


def save_metrics_table(methods, out_csv):
    rows = []
    for rec in methods:
        rows.append(
            {
                "Method": rec["Method"],
                "AUC": rec["AUC"],
                "ACC": rec["ACC"],
                "Precision": rec["Precision"],
                "Recall": rec["Recall"],
                "F1": rec["F1"],
                "MCC": rec["MCC"],
                "Threshold": "NA" if rec["threshold"] is None else rec["threshold"],
                "n_pos": rec["n_pos"],
                "n_neg": rec["n_neg"],
            }
        )
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df.to_csv(out_csv, index=False, na_rep="NA")
    return df


def plot_roc(methods, out_fig):
    if not methods:
        print("No valid results found.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    for i, rec in enumerate(methods):
        color = colors[i % len(colors)]
        ax.plot(
            rec["fpr"],
            rec["tpr"],
            color=color,
            linewidth=2.0,
            label=f"{rec['Method']} ({rec['AUC']:.2f})",
        )

    ax.plot([0, 1], [0, 1], color="0.7", linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=16)
    ax.set_ylabel("True Positive Rate", fontsize=16)
    ax.set_title("ROC Curves On Testing Set", fontsize=18, fontweight="bold", pad=10)

    for spine in ax.spines.values():
        spine.set_visible(True)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, fontsize=10, loc="lower right")
    ax.tick_params(direction="out", length=4, width=1.2)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    #plt.show()
    print(f"Figure saved: {out_fig}")
    print("Note: Due to method-specific limitations, such as sequence-length restrictions, missing GO annotations, or other required input constraints, some methods did not return scores for a small number of proteins.")

def format_value(value):
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}"
    return str(value)


def print_metrics_table(df):
    display_cols = ["Method", "AUC", "ACC", "Precision", "Recall", "F1", "MCC", "Threshold", "n_pos", "n_neg"]
    display_df = df[display_cols].copy()
    for col in ["AUC", "ACC", "Precision", "Recall", "F1", "MCC"]:
        display_df[col] = display_df[col].map(format_value)
    print(display_df.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pscore_threshold", type=float, default=4.0)
    parser.add_argument("--out_csv", default="metrics_summary.csv")
    parser.add_argument("--out_fig", default="ROC_curves.png")
    return parser.parse_args()


def main():
    args = parse_args()
    methods = collect_methods_from_subdirs(args.threshold, args.pscore_threshold)
    df = save_metrics_table(methods, args.out_csv)

    if len(df) == 0:
        print("No valid metrics were generated.")
    else:
        print_metrics_table(df)
        print(f"Metrics saved: {args.out_csv}")

    plot_roc(methods, args.out_fig)


if __name__ == "__main__":
    main()
