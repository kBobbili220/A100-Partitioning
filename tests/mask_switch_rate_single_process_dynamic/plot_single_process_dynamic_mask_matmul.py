#!/usr/bin/env python3
"""
Plots for the dynamic-mask sliced GEMM benchmark.

CSV rows are per-slice deploy (iter = slice index): mask schedule, host timings
(switch_us, launch_us, set_plus_launch_us), and optional SM residency samples.
"""
import csv
import os
import sys

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.ticker import (
    AutoMinorLocator,
    FormatStrFormatter,
    MaxNLocator,
    MultipleLocator,
)
import numpy as np

# Histogram bin width in microseconds (tenths of a µs).
HIST_BIN_WIDTH_US = 0.1


def load_iterations(path):
    rows = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                {
                    "iter": int(row["iter"]),
                    "mask_id": int(row["mask_id"]),
                    "expected_sm_lo": int(row["expected_sm_lo"]),
                    "expected_sm_hi": int(row["expected_sm_hi"]),
                    "dominant_tpc": int(row["dominant_tpc"]),
                    "switch_us": float(row["switch_us"]),
                    "launch_us": float(row["launch_us"]),
                    "set_plus_launch_us": float(row["set_plus_launch_us"]),
                    "sampled": int(row["sampled"]),
                    "inside_pct": float(row["inside_pct"]),
                    "outside_pct": float(row["outside_pct"]),
                    "smid_min": int(row["smid_min"]),
                    "smid_max": int(row["smid_max"]),
                    "distinct_smid_count": int(row["distinct_smid_count"]),
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


def rolling_mean(values, win):
    if len(values) == 0:
        return np.array([])
    w = max(1, min(win, len(values)))
    kernel = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(np.array(values, dtype=np.float64), kernel, mode="same")


def _style_latency_axis_us(ax, axis="y"):
    """Major/minor ticks: 0.1 µs minors when span is modest; coarser minors if span is large."""
    which = ax.yaxis if axis == "y" else ax.xaxis
    lim = ax.get_ylim() if axis == "y" else ax.get_xlim()
    span = max(float(lim[1]) - float(lim[0]), 1e-9)
    which.set_major_formatter(FormatStrFormatter("%.1f"))
    # Cap minor tick count: 0.1 µs step only when span is small enough for Matplotlib limits.
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
    """Readable x-axis for latency histograms: bounded tick count + tilted labels."""
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.xaxis.set_major_locator(
        MaxNLocator(nbins=max_major_ticks, min_n_ticks=4, steps=[1, 2, 2.5, 5, 10])
    )
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.tick_params(axis="x", labelsize=9)
    for lab in ax.get_xticklabels():
        lab.set_rotation(40)
        lab.set_ha("right")


def _deploy_ylim_us(*series):
    """Upper y-limit from bulk of data so rare spikes do not compress the trace."""
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


def _hist_bin_edges_us(values, bin_width_us=HIST_BIN_WIDTH_US):
    """Edges [0, w, 2w, ...] covering max(values); constant-width bins in µs."""
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
            "Usage: python3 plot_single_process_dynamic_mask_matmul.py "
            "<iterations.csv> <summary.csv>"
        )
        sys.exit(1)

    iter_csv = sys.argv[1]
    sum_csv = sys.argv[2]
    rows = load_iterations(iter_csv)
    summary = load_summary(sum_csv)
    base = os.path.splitext(iter_csv)[0]

    it = [r["iter"] for r in rows]
    mask_ids = [r["mask_id"] for r in rows]
    exp_lo = [r["expected_sm_lo"] for r in rows]
    exp_hi = [r["expected_sm_hi"] for r in rows]
    sampled = [r["sampled"] == 1 for r in rows]
    obs_lo = [r["smid_min"] for r in rows]
    obs_hi = [r["smid_max"] for r in rows]
    obs_lo_band = [obs_lo[i] if sampled[i] else np.nan for i in range(len(rows))]
    obs_hi_band = [obs_hi[i] if sampled[i] else np.nan for i in range(len(rows))]
    mismatch_indices = [i for i, r in enumerate(rows) if r["sampled"] == 1 and r["mismatch"] == 1]
    sw = [r["switch_us"] for r in rows]
    launch = [r["launch_us"] for r in rows]
    total = [r["set_plus_launch_us"] for r in rows]

    n_masks = max(mask_ids) + 1 if mask_ids else 1
    n_colors = max(n_masks, 3)
    try:
        cmap = mpl.colormaps["tab10"].resampled(n_colors)
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap("tab10", n_colors)

    # Residency: expected bounds vs observed SM band (vertical per slice); color by mask_id.
    plt.figure(figsize=(14, 5))
    ax = plt.gca()
    for i, x in enumerate(it):
        if not sampled[i]:
            continue
        lo = obs_lo_band[i]
        hi = obs_hi_band[i]
        if np.isnan(lo) or np.isnan(hi):
            continue
        y0 = min(lo, hi)
        y1 = max(lo, hi)
        color = cmap(mask_ids[i] % n_colors)
        ax.fill_betweenx([y0, y1], x - 0.5, x + 0.5, color=color, alpha=0.35, linewidth=0, zorder=2)
    h_exp_lo = plt.scatter(it, exp_lo, s=14, label="expected_sm_lo", c="black", alpha=0.85, zorder=4)
    h_exp_hi = plt.scatter(it, exp_hi, s=14, label="expected_sm_hi", c="dimgray", alpha=0.85, zorder=4)
    h_mismatch = None
    if mismatch_indices:
        mismatch_x = [rows[i]["iter"] for i in mismatch_indices]
        mismatch_y = [rows[i]["smid_max"] for i in mismatch_indices]
        h_mismatch = plt.scatter(
            mismatch_x,
            mismatch_y,
            s=28,
            marker="x",
            c="red",
            label="mismatch",
            zorder=5,
        )
    plt.xlabel("Slice index (deploy order)")
    plt.ylabel("SM ID")
    plt.title("Residency: expected SM bounds vs observed SM range (colored by mask_id)")
    plt.grid(alpha=0.25)
    handles = [
        mpatches.Patch(facecolor=cmap(0), alpha=0.35, edgecolor="none", label="observed band (by mask_id)"),
        h_exp_lo,
        h_exp_hi,
    ]
    if h_mismatch is not None:
        handles.append(h_mismatch)
    plt.legend(handles=handles, loc="best")
    plt.tight_layout()
    out_timeline = base + "_residency_timeline.png"
    plt.savefig(out_timeline, dpi=140)
    plt.close()

    # Host deploy path: mask API vs stream-create+enqueue vs full enqueue window.
    fig, ax = plt.subplots(figsize=(14, 5))
    rm_total = rolling_mean(np.array(total, dtype=np.float64), max(5, len(total) // 25))
    ax.plot(it, sw, linewidth=0.85, alpha=0.85, label="switch_us (mask API)")
    ax.plot(it, launch, linewidth=0.85, alpha=0.85, label="launch_us (create→enqueue)")
    ax.plot(it, total, linewidth=0.85, alpha=0.8, label="set_plus_launch_us (create→post-enqueue)")
    ax.plot(
        it,
        rm_total,
        linewidth=1.6,
        label="set_plus_launch_us rolling mean",
    )
    y0, y1 = _deploy_ylim_us(sw, launch, total, rm_total)
    ax.set_ylim(y0, y1)
    _style_latency_axis_us(ax, axis="y")
    ax.set_xlabel("Slice index")
    ax.set_ylabel("Latency (µs)")
    ax.set_title(
        "Per-slice host deploy latency (y-axis zoomed to ~p99.9 if spikes present; "
        "0.1 µs minor grid; overlapped GPU work; no GPU completion)"
    )
    ax.grid(which="major", alpha=0.35)
    ax.grid(which="minor", alpha=0.12)
    ax.legend(loc="best")
    plt.tight_layout()
    out_deploy = base + "_deploy_latency_us.png"
    plt.savefig(out_deploy, dpi=140)
    plt.close()

    # Distributions: mask API vs full deploy enqueue (0.1 µs-wide bins on x).
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sw_arr = np.array(sw, dtype=np.float64)
    total_arr = np.array(total, dtype=np.float64)
    if sw_arr.size > 0:
        edges_sw = _hist_bin_edges_us(sw_arr)
        axes[0].hist(sw_arr, bins=edges_sw, alpha=0.88, color="C0", edgecolor="none")
    axes[0].set_xlabel("switch_us (µs)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Mask API latency ({HIST_BIN_WIDTH_US:g} µs bins)")
    axes[0].grid(which="major", alpha=0.35)
    axes[0].grid(which="minor", alpha=0.12)
    _style_histogram_x_us(axes[0])
    if total_arr.size > 0:
        edges_t = _hist_bin_edges_us(total_arr)
        axes[1].hist(total_arr, bins=edges_t, alpha=0.88, color="C1", edgecolor="none")
    axes[1].set_xlabel("set_plus_launch_us (µs)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Full host enqueue window ({HIST_BIN_WIDTH_US:g} µs bins)")
    axes[1].grid(which="major", alpha=0.35)
    axes[1].grid(which="minor", alpha=0.12)
    _style_histogram_x_us(axes[1])
    plt.tight_layout(rect=(0, 0.11, 1, 1))
    out_hists = base + "_latency_hists.png"
    plt.savefig(out_hists, dpi=140)
    plt.close()

    print(f"Wrote: {out_timeline}")
    print(f"Wrote: {out_deploy}")
    print(f"Wrote: {out_hists}")
    if "switches_per_sec" in summary:
        print(f"switches_per_sec={summary['switches_per_sec']}")
    if "mismatch_count" in summary:
        print(f"mismatch_count={summary['mismatch_count']}")
    if "slice_rows" in summary:
        print(f"slice_rows={summary['slice_rows']}")
    if "max_inflight" in summary:
        print(f"max_inflight={summary['max_inflight']}")


if __name__ == "__main__":
    main()
