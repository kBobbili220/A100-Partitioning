#!/usr/bin/env python3
"""
Plot libsmctrl dynamic switching timeline CSV produced by:
  ./libsmctrl_dynamic_switch_demo ...

Outputs:
  - <input>_heatmap.png : time-window x SMID heatmap of block occupancy
  - <input>_durations.png : per-window kernel duration with phase coloring
"""

import csv
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    # Data keyed by window then smid
    occ = defaultdict(dict)
    meta = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            w = int(row["window"])
            sm = int(row["smid"])
            blocks = int(row["blocks"])
            occ[w][sm] = blocks
            meta[w] = {
                "phase": int(row["phase"]),
                "enabled_tpc": int(row["enabled_tpc"]),
                "duration_ms": float(row["duration_ms"]),
                "start_ns": int(row["start_ns"]),
                "end_ns": int(row["end_ns"]),
            }
    windows = sorted(meta.keys())
    max_sm = 0
    for w in windows:
        if occ[w]:
            max_sm = max(max_sm, max(occ[w].keys()))
    num_sms = max_sm + 1

    mat = np.zeros((len(windows), num_sms), dtype=np.int32)
    phase = np.zeros((len(windows),), dtype=np.int32)
    tpc = np.zeros((len(windows),), dtype=np.int32)
    durations = np.zeros((len(windows),), dtype=np.float64)
    for i, w in enumerate(windows):
        phase[i] = meta[w]["phase"]
        tpc[i] = meta[w]["enabled_tpc"]
        durations[i] = meta[w]["duration_ms"]
        for sm, blocks in occ[w].items():
            mat[i, sm] = blocks
    return mat, phase, tpc, durations


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_dynamic_switch_timeline.py <timeline.csv>")
        sys.exit(1)

    in_csv = sys.argv[1]
    mat, phase, tpc, durations = load_csv(in_csv)
    base = os.path.splitext(in_csv)[0]

    # Heatmap
    plt.figure(figsize=(16, 6))
    plt.imshow(mat, aspect="auto", interpolation="nearest", cmap="viridis")
    plt.colorbar(label="Blocks observed on SM in window")
    plt.title("Dynamic SM Allocation Timeline (window x SMID)")
    plt.xlabel("SMID")
    plt.ylabel("Window")
    # Mark phase boundaries where TPC assignment can flip.
    for i in range(1, len(phase)):
        if phase[i] != phase[i - 1]:
            plt.axhline(i - 0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.8)
    plt.tight_layout()
    out_heat = base + "_heatmap.png"
    plt.savefig(out_heat, dpi=140)
    plt.close()

    # Duration plot
    plt.figure(figsize=(16, 4))
    x = np.arange(len(durations))
    uniq_phases = np.unique(phase)
    for p in uniq_phases:
        idx = np.where(phase == p)[0]
        plt.scatter(idx, durations[idx], s=20, label=f"phase {p} (TPC {tpc[idx[0]]})")
    plt.plot(x, durations, linewidth=0.8, alpha=0.7)
    plt.title("Per-window Kernel Duration")
    plt.xlabel("Window")
    plt.ylabel("Duration (ms)")
    plt.grid(alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    out_dur = base + "_durations.png"
    plt.savefig(out_dur, dpi=140)
    plt.close()

    print(f"Wrote: {out_heat}")
    print(f"Wrote: {out_dur}")


if __name__ == "__main__":
    main()

