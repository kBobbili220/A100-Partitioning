#!/usr/bin/env python3
"""Plots for two-process alternating-mask sliced GEMM benchmark."""
import csv
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import (
    AutoMinorLocator,
    FormatStrFormatter,
    MaxNLocator,
    MultipleLocator,
)
import numpy as np

HIST_BIN_WIDTH_US = 0.1


def load_iterations(path):
    rows = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                {
                    "iter": int(row["iter"]),
                    "phase": int(row["phase"]),
                    "role": int(row["role"]),
                    "mask_id": int(row["mask_id"]),
                    "expected_sm_lo": int(row["expected_sm_lo"]),
                    "expected_sm_hi": int(row["expected_sm_hi"]),
                    "switch_us": float(row["switch_us"]),
                    "launch_us": float(row["launch_us"]),
                    "set_plus_launch_us": float(row["set_plus_launch_us"]),
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


def rolling_mean(values, win):
    if len(values) == 0:
        return np.array([])
    w = max(1, min(win, len(values)))
    kernel = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(np.array(values, dtype=np.float64), kernel, mode="same")


def _deploy_ylim_us(*series):
    parts = []
    for s in series:
        a = np.asarray(s, dtype=np.float64)
        a = a[np.isfinite(a) & (a >= 0)]
        if a.size:
            parts.append(a)
    if not parts:
        return 0.0, 1.0
    stacked = np.concatenate(parts)
    p999 = float(np.percentile(stacked, 99.9))
    mx = float(np.max(stacked))
    if mx <= p999 * 1.15:
        ymax = mx * 1.06
    else:
        ymax = p999 * 1.08
    ymax = max(ymax, 0.2)
    return 0.0, ymax


def _style_latency_axis_us(ax, axis="y"):
    which = ax.yaxis if axis == "y" else ax.xaxis
    lim = ax.get_ylim() if axis == "y" else ax.get_xlim()
    span = max(float(lim[1]) - float(lim[0]), 1e-9)
    which.set_major_formatter(FormatStrFormatter("%.1f"))
    if span <= 25.0:
        which.set_minor_locator(MultipleLocator(0.1))
    elif span <= 80.0:
        which.set_minor_locator(MultipleLocator(0.5))
    elif span <= 200.0:
        which.set_minor_locator(MultipleLocator(1.0))
    else:
        which.set_minor_locator(AutoMinorLocator(5))
    if span <= 1.5:
        which.set_major_locator(MultipleLocator(0.2))
    elif span <= 4.0:
        which.set_major_locator(MultipleLocator(0.5))
    elif span <= 12.0:
        which.set_major_locator(MultipleLocator(1.0))
    elif span <= 30.0:
        which.set_major_locator(MultipleLocator(2.0))
    else:
        which.set_major_locator(MultipleLocator(5.0))


def _style_histogram_x_us(ax, max_major_ticks=17):
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.xaxis.set_major_locator(
        MaxNLocator(nbins=max_major_ticks, min_n_ticks=4, steps=[1, 2, 2.5, 5, 10])
    )
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.tick_params(axis="x", labelsize=9)
    for lab in ax.get_xticklabels():
        lab.set_rotation(40)
        lab.set_ha("right")


def _hist_bin_edges_us(values, bin_width_us=HIST_BIN_WIDTH_US):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v) & (v >= 0)]
    if v.size == 0:
        return np.arange(0.0, 1.0 + bin_width_us, bin_width_us)
    mx = float(np.max(v))
    n = int(np.ceil(mx / bin_width_us)) + 1
    return np.arange(0.0, n * bin_width_us + 0.5 * bin_width_us, bin_width_us)


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python3 plot_two_process_dynamic_mask_matmul.py "
            "<iterations.csv> <summary.csv>"
        )
        sys.exit(1)

    iter_csv = sys.argv[1]
    sum_csv = sys.argv[2]
    rows = load_iterations(iter_csv)
    summary = load_summary(sum_csv)
    base = os.path.splitext(iter_csv)[0]

    it = np.array([r["iter"] for r in rows], dtype=np.int32)
    phase = np.array([r["phase"] for r in rows], dtype=np.int32)
    role = np.array([r["role"] for r in rows], dtype=np.int32)
    sw = np.array([r["switch_us"] for r in rows], dtype=np.float64)
    launch = np.array([r["launch_us"] for r in rows], dtype=np.float64)
    total = np.array([r["set_plus_launch_us"] for r in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(14, 5))
    for rid, label, color in ((0, "role A (parent)", "C0"), (1, "role B (child)", "C1")):
        m = role == rid
        if not np.any(m):
            continue
        ax.plot(it[m], sw[m], linewidth=0.85, alpha=0.85, color=color, linestyle="--", label=f"{label} switch_us")
        ax.plot(it[m], total[m], linewidth=0.9, alpha=0.88, color=color, label=f"{label} set_plus_launch_us")
    rm = rolling_mean(total, max(5, max(1, len(total) // 25)))
    rmv = rm[np.isfinite(rm)] if len(rm) else np.array([])
    if rmv.size:
        ax.plot(it, rm, linewidth=1.4, color="dimgray", label="set_plus_launch_us rolling mean (all rows)")
    y0, y1 = _deploy_ylim_us(sw, launch, total, rmv)
    ax.set_ylim(y0, y1)
    _style_latency_axis_us(ax, axis="y")
    ax.set_xlabel("Slice index (per process)")
    ax.set_ylabel("Latency (µs)")
    ax.set_title("Two-process deploy latency by role (y zoom ~p99.9; 0.1 µs minor grid)")
    ax.grid(which="major", alpha=0.35)
    ax.grid(which="minor", alpha=0.12)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    out_deploy = base + "_deploy_latency_us.png"
    plt.savefig(out_deploy, dpi=140)
    plt.close()

    if rows:
        fig, ax = plt.subplots(figsize=(12, 4))
        sc = ax.scatter(it, phase, c=role, cmap="coolwarm", s=14, alpha=0.8)
        ax.set_xlabel("Slice index")
        ax.set_ylabel("phase (0/1)")
        ax.set_title("Phase vs slice index (color: role 0=A, 1=B)")
        ax.grid(alpha=0.25)
        plt.colorbar(sc, ax=ax, label="role")
        plt.tight_layout()
        out_phase = base + "_phase_timeline.png"
        plt.savefig(out_phase, dpi=140)
        plt.close()
    else:
        out_phase = None

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    if sw.size > 0:
        edges = _hist_bin_edges_us(sw)
        axes[0].hist(sw, bins=edges, alpha=0.88, color="C2", edgecolor="none")
    axes[0].set_xlabel("switch_us (µs)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Mask API latency ({HIST_BIN_WIDTH_US:g} µs bins)")
    axes[0].grid(which="major", alpha=0.35)
    axes[0].grid(which="minor", alpha=0.12)
    _style_histogram_x_us(axes[0])
    if total.size > 0:
        edges_t = _hist_bin_edges_us(total)
        axes[1].hist(total, bins=edges_t, alpha=0.88, color="C3", edgecolor="none")
    axes[1].set_xlabel("set_plus_launch_us (µs)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Full enqueue window ({HIST_BIN_WIDTH_US:g} µs bins)")
    axes[1].grid(which="major", alpha=0.35)
    axes[1].grid(which="minor", alpha=0.12)
    _style_histogram_x_us(axes[1])
    plt.tight_layout(rect=(0, 0.11, 1, 1))
    out_hist = base + "_latency_hists.png"
    plt.savefig(out_hist, dpi=140)
    plt.close()

    print(f"Wrote: {out_deploy}")
    if out_phase:
        print(f"Wrote: {out_phase}")
    print(f"Wrote: {out_hist}")
    if "merged_rows_per_sec" in summary:
        print(f"merged_rows_per_sec={summary['merged_rows_per_sec']}")


if __name__ == "__main__":
    main()
