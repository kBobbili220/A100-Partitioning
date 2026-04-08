# Dual-TPC Concurrent GEMM Demo

Build from `libsmctrl/`:

`make libsmctrl_dual_tpc_gemm_demo`

Run example:

`../tests/parallel_partitioning/libsmctrl_dual_tpc_gemm_demo --tpcs-a 0-19 --tpcs-b 30-49 --m 4096 --n 4096 --k 4096 --iters 30 --repeat 2`

Recommended run with CSV output:

`../tests/parallel_partitioning/libsmctrl_dual_tpc_gemm_demo --tpcs-a 0-19 --tpcs-b 30-49 --m 4096 --n 4096 --k 4096 --iters 30 --repeat 2 --csv dual_tpc_gemm_metrics.csv`

Note: when `--csv` is relative (like above), output is written to the binary directory (`tests/parallel_partitioning`), not the current working directory.

Generate graphs:

`python3 ../tests/parallel_partitioning/plot_dual_tpc_gemm.py dual_tpc_gemm_metrics.csv`

Note: when the CSV argument is relative, the script resolves it relative to `tests/parallel_partitioning` and writes plots there as well.

Generated plots:
- `*_sm_heatmap.png`: side-by-side SM occupancy for workloads A and B
- `*_tpc_hist.png`: inferred per-TPC usage bars for A vs B
- `*_leakage.png`: runtime/spread summary

Terminal output is now concise:
- run configuration and process attribution (`pid`, stream handles)
- leakage percentages and final `PASS/WARN`
- CSV/plot command hints
