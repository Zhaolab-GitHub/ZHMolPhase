#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np


COLOR_PALETTE = [
    "#fa7f6f",
    "#ffbe7a",
    "#cde58b",
    "#82afda",
    "#e8a9d0",
    "#f2a7ad",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot success ratios from result.txt.")
    parser.add_argument("--input", default="all_result.txt", help="Input result file.")
    parser.add_argument("--out_fig", default="result_single_group.png", help="Output figure path.")
    parser.add_argument("--out_csv", default="result_summary.csv", help="Output summary CSV path.")
    parser.add_argument("--show", action="store_true", help="Display the figure interactively.")
    return parser.parse_args()


def read_results(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    records = []
    pattern = re.compile(r"^(.+?)\s+(\d+)\s*/\s*(\d+)\s*$")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            match = pattern.match(line)
            if match is None:
                print(f"Skipped unparsable line {line_no}: {line}", file=sys.stderr)
                continue
            name, success, total = match.groups()
            success = int(success)
            total = int(total)
            ratio = success / total if total > 0 else 0.0
            records.append(
                {
                    "name": name.strip().strip('"').strip("'"),
                    "success": success,
                    "total": total,
                    "ratio": ratio,
                    "label": f"{success}/{total}",
                }
            )

    if not records:
        raise ValueError(f"No valid records found in: {path}")

    return records


def write_summary(records, out_csv):
    total_success = sum(r["success"] for r in records)
    total_count = sum(r["total"] for r in records)
    overall_ratio = total_success / total_count if total_count > 0 else 0.0

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Success", "Total", "Success_Ratio"])
        for r in records:
            writer.writerow([r["name"], r["success"], r["total"], f"{r['ratio']:.6f}"])
        writer.writerow(["Overall", total_success, total_count, f"{overall_ratio:.6f}"])

    print("Method\tSuccess\tTotal\tSuccess_Ratio")
    for r in records:
        print(f"{r['name']}\t{r['success']}\t{r['total']}\t{r['ratio']:.6f}")
    print(f"Overall\t{total_success}\t{total_count}\t{overall_ratio:.6f}")
    print(f"Saved summary: {out_csv}")


def plot_results(records, out_fig, show=False):
    names = np.array([r["name"] for r in records])
    ratios = np.array([r["ratio"] for r in records], dtype=float)
    labels = [f"{r['success']}/{r['total']}\n{r['ratio'] * 100:.1f}%" for r in records]
    colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(len(records))]

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    x = np.arange(len(names))
    bars = ax.bar(x, ratios, width=0.8, color=colors, alpha=0.9)

    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylabel("Frequency of correct predictions", fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="center", fontsize=16)

    for bar, label in zip(bars, labels):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            label,
            ha="center",
            va="bottom",
            fontsize=12,
        )

    fig.tight_layout()
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {out_fig}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    args = parse_args()
    records = read_results(args.input)
    write_summary(records, args.out_csv)
    plot_results(records, args.out_fig, args.show)


if __name__ == "__main__":
    main()
