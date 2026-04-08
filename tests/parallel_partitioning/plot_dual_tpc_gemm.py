#!/usr/bin/env python3
"""
Plot CSV output from libsmctrl_dual_tpc_gemm_demo.

Outputs:
  - <input>_sm_heatmap.png
  - <input>_tpc_hist.png
  - <input>_leakage.png
"""

import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


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
    elapsed_ms = {"A": 0.0, "B": 0.0}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            w = row["workload"]
            kind = row["kind"]
            elapsed_ms[w] = float(row["elapsed_ms"])
            blocks = int(row["blocks"])
            if kind == "sm_hist":
                smid = int(row["smid"])
                sm[w][smid] = blocks
            elif kind == "tpc_hist":
                tpcid = int(row["tpc"])
                tpc[w][tpcid] = blocks

    max_sm = max(max(sm["A"].keys(), default=0), max(sm["B"].keys(), default=0))
    max_tpc = max(max(tpc["A"].keys(), default=0), max(tpc["B"].keys(), default=0))
    sm_mat = np.zeros((2, max_sm + 1), dtype=np.int32)
    tpc_mat = np.zeros((2, max_tpc + 1), dtype=np.int32)
    for widx, w in enumerate(["A", "B"]):
        for smid, blocks in sm[w].items():
            sm_mat[widx, smid] = blocks
        for tpcid, blocks in tpc[w].items():
            tpc_mat[widx, tpcid] = blocks
    return sm_mat, tpc_mat, elapsed_ms


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
    sm_mat, tpc_mat, elapsed_ms = load_csv(in_csv)
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

    print(f"Wrote: {out_sm}")
    print(f"Wrote: {out_tpc}")
    print(f"Wrote: {out_leak}")
    # Totals help quickly sanity-check if one workload issued/finished
    # substantially different CTA volume than the other in the measurement window.
    print(f"Totals: A_blocks={int(total_a)} B_blocks={int(total_b)}")


if __name__ == "__main__":
    main()

