#!/usr/bin/env python3
"""
Plot CSV output from libsmctrl_dual_tpc_gemm_demo.

Outputs:
  - <input>_sm_heatmap.png
  - <input>_tpc_hist.png
  - <input>_leakage.png
  - <input>_tpc_timeline.png
"""

import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


# Parse the benchmark CSV into plotting matrices.
#
# Input is a long-form table with two record types:
# - sm_hist: per-workload block counts per SM
# - tpc_hist: per-workload block counts per inferred TPC
#
# Output:
# - sm_mat[2, num_sms]
# - tpc_mat[2, num_tpcs]
# - elapsed_ms dict for A and B
def load_csv(path):
    sm = {"A": {}, "B": {}}
    tpc = {"A": {}, "B": {}}
    tpc_set = {"A": set(), "B": set()}
    intervals = {"A": [], "B": []}
    elapsed_ms = {"A": 0.0, "B": 0.0}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            w = row["workload"]
            kind = row["kind"]
            blocks = int(row["blocks"])
            if kind == "sm_hist":
                elapsed_ms[w] = float(row["elapsed_ms"])
                smid = int(row["smid"])
                sm[w][smid] = blocks
            elif kind == "tpc_hist":
                elapsed_ms[w] = float(row["elapsed_ms"])
                tpcid = int(row["tpc"])
                tpc[w][tpcid] = blocks
            elif kind == "tpc_set":
                elapsed_ms[w] = float(row["elapsed_ms"])
                tpc_set[w].add(int(row["tpc"]))
            elif kind == "kernel_interval":
                start_us = float(row["elapsed_ms"])
                end_us = float(row["blocks"])
                if end_us >= start_us:
                    intervals[w].append((start_us / 1000.0, end_us / 1000.0))

    max_sm = max(max(sm["A"].keys(), default=0), max(sm["B"].keys(), default=0))
    max_tpc = max(max(tpc["A"].keys(), default=0), max(tpc["B"].keys(), default=0))
    sm_mat = np.zeros((2, max_sm + 1), dtype=np.int32)
    tpc_mat = np.zeros((2, max_tpc + 1), dtype=np.int32)
    for widx, w in enumerate(["A", "B"]):
        for smid, blocks in sm[w].items():
            sm_mat[widx, smid] = blocks
        for tpcid, blocks in tpc[w].items():
            tpc_mat[widx, tpcid] = blocks
    intervals["A"].sort(key=lambda x: x[0])
    intervals["B"].sort(key=lambda x: x[0])
    return sm_mat, tpc_mat, tpc_set, intervals, elapsed_ms


# Resolve relative CSV paths against this script's directory so users can run
# from any CWD and still target tests/parallel_partitioning artifacts.
def resolve_input_path(arg_path):
    if os.path.isabs(arg_path):
        return arg_path
    # Anchor relative paths to this script's directory so plots land in tests/parallel_partitioning.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, arg_path)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_dual_tpc_gemm.py <metrics.csv>")
        sys.exit(1)

    in_csv = resolve_input_path(sys.argv[1])
    sm_mat, tpc_mat, tpc_set, intervals, elapsed_ms = load_csv(in_csv)
    base = os.path.splitext(in_csv)[0]

    # 1) SM heatmap
    # Rows are workloads (A/B), columns are SM IDs, values are observed block
    # counts. This is the most direct view of where CTAs were placed.
    plt.figure(figsize=(14, 3))
    plt.imshow(sm_mat, aspect="auto", interpolation="nearest", cmap="viridis")
    plt.colorbar(label="Observed blocks")
    plt.yticks([0, 1], ["Workload A", "Workload B"])
    plt.xlabel("SMID")
    plt.title("Per-Workload SM Occupancy")
    plt.tight_layout()
    out_sm = base + "_sm_heatmap.png"
    plt.savefig(out_sm, dpi=140)
    plt.close()

    # 2) TPC histogram (grouped bars)
    # Aggregates SM-level counts into inferred TPC bins to visualize whether A
    # and B stayed inside their intended partitions.
    x = np.arange(tpc_mat.shape[1])
    width = 0.42
    plt.figure(figsize=(14, 4))
    plt.bar(x - width / 2, tpc_mat[0], width=width, label="Workload A")
    plt.bar(x + width / 2, tpc_mat[1], width=width, label="Workload B")
    plt.xlabel("TPC")
    plt.ylabel("Observed blocks")
    plt.title("Per-Workload Inferred TPC Usage")
    plt.legend(loc="best")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out_tpc = base + "_tpc_hist.png"
    plt.savefig(out_tpc, dpi=140)
    plt.close()

    # 3) Runtime/spread summary chart
    # This plot is intentionally lightweight: elapsed time and the number of
    # active TPC bins ("spread") for A/B. Exact leakage percentages are computed
    # and printed by the C++ runner.
    total_a = float(np.sum(tpc_mat[0]))
    total_b = float(np.sum(tpc_mat[1]))
    active_a = int(np.count_nonzero(tpc_mat[0]))
    active_b = int(np.count_nonzero(tpc_mat[1]))
    vals = [elapsed_ms["A"], elapsed_ms["B"], active_a, active_b]
    labels = ["A elapsed ms", "B elapsed ms", "A active TPCs", "B active TPCs"]
    plt.figure(figsize=(9, 4))
    plt.bar(labels, vals)
    plt.title("Runtime and Spread Summary")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out_leak = base + "_leakage.png"
    plt.savefig(out_leak, dpi=140)
    plt.close()

    # 4) TPC-vs-time timeline.
    # Each pixel/bin indicates which workload was active on a TPC at that time:
    # 0=idle, 1=A-only, 2=B-only, 3=both (overlap). This gives a direct visual
    # proof of concurrent execution windows.
    all_tpcs = list(range(tpc_mat.shape[1]))
    if all_tpcs:
        row_of_tpc = {tpc_id: idx for idx, tpc_id in enumerate(all_tpcs)}
        time_end = 0.0
        for w in ["A", "B"]:
            for _, end in intervals[w]:
                time_end = max(time_end, end)
            time_end = max(time_end, elapsed_ms[w])
        n_bins = max(600, len(intervals["A"]) * 12, len(intervals["B"]) * 12)
        t_edges = np.linspace(0.0, max(1e-6, time_end), n_bins + 1)
        timeline = np.zeros((len(all_tpcs), n_bins), dtype=np.uint8)

        def mark(workload_code, workload_name):
            allowed = tpc_set[workload_name]
            if not allowed:
                return
            for start, end in intervals[workload_name]:
                if end <= start:
                    continue
                left = max(0, np.searchsorted(t_edges, start, side="right") - 1)
                right = min(n_bins, np.searchsorted(t_edges, end, side="left"))
                if right <= left:
                    right = min(n_bins, left + 1)
                for tpc_id in allowed:
                    row = row_of_tpc.get(tpc_id)
                    if row is not None:
                        timeline[row, left:right] |= workload_code

        mark(1, "A")
        mark(2, "B")

        plt.figure(figsize=(14, max(4.5, 0.1 * len(all_tpcs))))
        cmap = ListedColormap(["#101010", "#d62728", "#1f77b4", "#f2c744"])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
        plt.imshow(
            timeline,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
            extent=[0.0, max(1e-6, time_end), -0.5, len(all_tpcs) - 0.5],
            origin="lower",
        )
        cbar = plt.colorbar(ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(["Idle", "A running", "B running", "A+B overlap"])
        plt.xlabel("Time (ms)")
        plt.ylabel("TPC")
        if len(all_tpcs) <= 40:
            yticks = np.arange(len(all_tpcs))
        else:
            step = max(1, len(all_tpcs) // 20)
            yticks = np.arange(0, len(all_tpcs), step)
            if yticks[-1] != len(all_tpcs) - 1:
                yticks = np.append(yticks, len(all_tpcs) - 1)
        plt.yticks(yticks, [str(int(t)) for t in yticks])
        plt.title("TPC Activity Timeline (Program per TPC over time)")
        plt.tight_layout()
        out_timeline = base + "_tpc_timeline.png"
        plt.savefig(out_timeline, dpi=140)
        plt.close()
    else:
        out_timeline = base + "_tpc_timeline.png"

    print(f"Wrote: {out_sm}")
    print(f"Wrote: {out_tpc}")
    print(f"Wrote: {out_leak}")
    print(f"Wrote: {out_timeline}")
    if all_tpcs:
        state_counts = np.bincount(timeline.ravel(), minlength=4)
        print(
            "Timeline bins: "
            f"idle={int(state_counts[0])} "
            f"A={int(state_counts[1])} "
            f"B={int(state_counts[2])} "
            f"overlap={int(state_counts[3])}"
        )
    # Totals help quickly sanity-check if one workload issued/finished
    # substantially different CTA volume than the other in the measurement window.
    print(f"Totals: A_blocks={int(total_a)} B_blocks={int(total_b)}")


if __name__ == "__main__":
    main()

