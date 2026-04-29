# Two-Process Alternating SM-Mask Benchmark

This benchmark measures **how fast two OS processes can alternate `libsmctrl_set_stream_mask_ext` on their own CUDA streams** while each advances an **independent sliced GEMM** (`C = A * B`). The two processes use **two fixed, disjoint SM partitions** (mapped to TPC disable masks); each phase they **swap** which partition each process is allowed to use, so the device is always split between them without overlapping allowed TPCs.

Goals:

- **A)** Stress multiproc **mask + enqueue** cadence when both processes flip masks in lockstep.
- **B)** With `--drain-before-swap`, approximate **clean handoff** semantics: both synchronize their streams after each slice before the next barrier, so the next mask is applied after queued work on that stream has completed.

## Why not arbitrary overlapping swaps?

Stream masks only affect **where new kernels from that stream may run**; they do not evict other processes’ warps. Swapping to ranges that overlap what the peer still has in flight blurs “who owns which SM” at a given instant. The **partition-swap** pattern keeps two complementary masks and only exchanges **which process holds which mask**.

Partitions must be **disjoint in TPC enable bits** after SM→TPC mapping (same heuristic as the single-process benchmark: `SM / (num_sms / num_tpcs)`). Example for a typical A100-style layout with two SMs per TPC: use non-overlapping SM bands such as **`0-41`** and **`42-107`** so the TPC sets do not overlap.

## CUDA and `fork`

The binary **`fork()`s before any CUDA initialization**. Each process then calls `cudaSetDevice`, allocates its own `A,B,C` buffers, and runs the measured loop. **Do not** initialize CUDA in the parent before `fork` and then use CUDA in both parent and child without the usual constraints—this program follows the safe pattern.

## Build

From `A100-partitioning/libsmctrl`:

`make libsmctrl_two_process_dynamic_mask_matmul`

## Run

From `A100-partitioning/libsmctrl`:

`../tests/mask_switch_rate_two_process_dynamic/libsmctrl_two_process_dynamic_mask_matmul --partition0 0-41 --partition1 42-107 --slice-rows 256 --max-phases 1000000 --warmup 5 --m 2048 --n 2048 --k 2048 --csv-prefix two_process_dynamic_mask_matmul`

Useful knobs:

- `--partition0 lo-hi` / `--partition1 lo-hi`: disjoint SM ranges (required).
- `--max-phases`: cap phase iterations per process (each phase advances one slice of `slice_rows` rows).
- `--slice-rows`: rows per slice.
- `--m --n --k`: GEMM shape (each process runs a full GEMM of this size).
- `--device N`: CUDA device index.
- `--drain-before-swap`: `cudaStreamSynchronize` after each slice before the end-of-phase barrier.
- `--csv-prefix`: output basename next to the executable.

## Outputs

Written next to the executable in `tests/mask_switch_rate_two_process_dynamic`:

- `<prefix>_iterations_A.csv` — parent (role 0) rows
- `<prefix>_iterations_B.csv` — child (role 1) rows
- `<prefix>_iterations.csv` — merged (sorted by `iter`, then `role`)
- `<prefix>_summary.csv` — aggregate stats over merged rows + wall-clock throughput

CSV columns (iterations): `iter,phase,role,mask_id,expected_sm_lo,expected_sm_hi,switch_us,launch_us,set_plus_launch_us`

- `phase`: `iter % 2` (which side of the swap).
- `role`: `0` = process A (parent), `1` = process B (child).
- `mask_id`: `0` if this row used `--partition0`, else `1` for `--partition1`.

## Plot

From `A100-partitioning/libsmctrl`:

`python3 ../tests/mask_switch_rate_two_process_dynamic/plot_two_process_dynamic_mask_matmul.py ../tests/mask_switch_rate_two_process_dynamic/two_process_dynamic_mask_matmul_iterations.csv ../tests/mask_switch_rate_two_process_dynamic/two_process_dynamic_mask_matmul_summary.csv`

Produces:

- `<prefix>_iterations_deploy_latency_us.png` — per-role `switch_us` and `set_plus_launch_us` vs slice index
- `<prefix>_iterations_phase_timeline.png` — `phase` vs `iter`, colored by `role`
- `<prefix>_iterations_latency_hists.png` — histograms of mask API vs full enqueue window

## Metric notes

- `switch_us`: host time in `libsmctrl_set_stream_mask_ext` only.
- `launch_us`: host time from after mask set through kernel launch enqueue.
- `set_plus_launch_us`: host time from before mask set through launch enqueue (same window style as the single-process dynamic benchmark).
- Timings are **host-side**; without `--drain-before-swap`, GPU work from prior phases may still be in flight.

Synchronization uses **`pthread_barrier_t`** in **`mmap(MAP_SHARED)`** so both processes rendezvous at the start and end of each phase.
