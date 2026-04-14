#!/usr/bin/env python3
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def load_iterations(path):
    rows = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                {
                    "iter": int(row["iter"]),
                    "requested_tpc": int(row["requested_tpc"]),
                    "dominant_tpc": int(row["dominant_tpc"]),
                    "switch_us": float(row["switch_us"]),
                    "cycle_us": float(row["cycle_us"]),
                    "outside_pct": float(row["outside_pct"]),
                    "mismatch": int(row["mismatch"]),
                }
            )
    return rows


def load_summary(path):
    out = {}
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out[row["metric"]] = row["value"]
    return out


def rolling_mean(x, win):
    if len(x) == 0:
        return np.array([])
    w = max(1, min(win, len(x)))
    kernel = np.ones(w) / float(w)
    return np.convolve(np.array(x, dtype=np.float64), kernel, mode="same")


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 plot_mask_switch_rate.py <iterations.csv> <summary.csv>")
        sys.exit(1)

    iter_csv = sys.argv[1]
    sum_csv = sys.argv[2]
    rows = load_iterations(iter_csv)
    summary = load_summary(sum_csv)
    base = os.path.splitext(iter_csv)[0]

    it = [r["iter"] for r in rows]
    sw = [r["switch_us"] for r in rows]
    cy = [r["cycle_us"] for r in rows]
    req = [r["requested_tpc"] for r in rows]
    dom = [r["dominant_tpc"] for r in rows]
    mismatch_idx = [r["iter"] for r in rows if r["mismatch"] == 1]

    # 1) switch overhead time series
    plt.figure(figsize=(14, 4))
    plt.plot(it, sw, linewidth=0.9, alpha=0.8, label="switch_us")
    plt.plot(it, rolling_mean(sw, max(5, len(sw) // 20)), linewidth=1.5, label="rolling mean")
    plt.xlabel("Iteration")
    plt.ylabel("Switch overhead (us)")
    plt.title("Per-Iteration SM Mask Switch Overhead")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    out1 = base + "_switch_us.png"
    plt.savefig(out1, dpi=140)
    plt.close()

    # 2) end-to-end cycle time
    plt.figure(figsize=(14, 4))
    plt.plot(it, cy, linewidth=0.9, alpha=0.8)
    plt.xlabel("Iteration")
    plt.ylabel("Cycle time (us)")
    plt.title("Per-Iteration Cycle Time (mask set + launch + sync)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out2 = base + "_cycle_us.png"
    plt.savefig(out2, dpi=140)
    plt.close()

    # 3) requested vs observed dominant TPC
    plt.figure(figsize=(14, 4))
    plt.plot(it, req, linewidth=1.0, label="requested_tpc")
    plt.plot(it, dom, linewidth=1.0, label="dominant_observed_tpc")
    if mismatch_idx:
        y = [dom[i] for i in mismatch_idx]
        plt.scatter(mismatch_idx, y, s=16, marker="x", label="mismatch")
    plt.xlabel("Iteration")
    plt.ylabel("TPC")
    plt.title("Requested vs Observed Dominant TPC")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    out3 = base + "_tpc_match.png"
    plt.savefig(out3, dpi=140)
    plt.close()

    print(f"Wrote: {out1}")
    print(f"Wrote: {out2}")
    print(f"Wrote: {out3}")
    if "switches_per_sec" in summary:
        print(f"switches_per_sec={summary['switches_per_sec']}")
    if "mismatch_count" in summary:
        print(f"mismatch_count={summary['mismatch_count']}")


if __name__ == "__main__":
    main()

