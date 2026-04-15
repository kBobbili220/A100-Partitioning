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
                    "switch_us": float(row["switch_us"]),
                    "launch_us": float(row["launch_us"]),
                    "set_and_launch_us": float(row["set_and_launch_us"]),
                    "iter_t0_ns": int(row["iter_t0_ns"]),
                    "iter_t1_ns": int(row["iter_t1_ns"]),
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
        print("Usage: python3 plot_dual_process_mask_check.py <iterations.csv> <summary.csv>")
        sys.exit(1)

    iter_csv = sys.argv[1]
    sum_csv = sys.argv[2]
    rows = load_iterations(iter_csv)
    summary = load_summary(sum_csv)
    base = os.path.splitext(iter_csv)[0]

    it = [r["iter"] for r in rows]
    sw = [r["switch_us"] for r in rows]
    launch = [r["launch_us"] for r in rows]
    total = [r["set_and_launch_us"] for r in rows]

    # 1) latency plot (switch, launch, and combined)
    plt.figure(figsize=(14, 4))
    plt.plot(it, sw, linewidth=0.8, alpha=0.7, label="switch_us")
    plt.plot(it, launch, linewidth=0.8, alpha=0.7, label="launch_us")
    plt.plot(it, total, linewidth=0.9, alpha=0.7, label="set_and_launch_us")
    plt.plot(
        it,
        rolling_mean(total, max(5, len(total) // 25)),
        linewidth=1.5,
        label="set_and_launch_us rolling mean",
    )
    plt.xlabel("Iteration")
    plt.ylabel("Latency (us)")
    plt.title("Mask Set + Kernel Deploy Latency Per Iteration")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    out1 = base + "_latency_us.png"
    plt.savefig(out1, dpi=140)
    plt.close()

    # 2) cycle-style iteration plot (keeps legacy output filename)
    plt.figure(figsize=(14, 4))
    plt.plot(it, total, linewidth=0.9, alpha=0.8)
    plt.xlabel("Iteration")
    plt.ylabel("set_and_launch_us")
    plt.title("Per-Iteration Cycle Time (mask set + launch)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out2 = base + "_throughput_lps.png"
    plt.savefig(out2, dpi=140)
    plt.close()

    # 3) distribution of combined set+launch latency
    plt.figure(figsize=(14, 4))
    if len(total) > 0:
        bins = min(200, max(30, len(total) // 50))
        plt.hist(np.array(total, dtype=np.float64), bins=bins, alpha=0.85)
    plt.xlabel("set_and_launch_us")
    plt.ylabel("Count")
    plt.title("Distribution of Mask Set + Kernel Deploy Latency")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out3 = base + "_set_and_launch_hist.png"
    plt.savefig(out3, dpi=140)
    plt.close()

    print(f"Wrote: {out1}")
    print(f"Wrote: {out2}")
    print(f"Wrote: {out3}")
    if "launches_per_sec" in summary:
        print(f"launches_per_sec={summary['launches_per_sec']}")


if __name__ == "__main__":
    main()
