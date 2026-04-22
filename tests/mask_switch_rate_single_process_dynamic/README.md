# Single-Process Dynamic SM-Mask Matmul Benchmark

This benchmark runs **one large GEMM** (`C = A * B`) by slicing work across output rows.
For each slice it **creates a fresh CUDA stream**, applies the next SM-range-derived TPC mask
via `libsmctrl_set_stream_mask_ext(...)`, enqueues the GEMM slice on that stream, records a
completion event, and immediately continues enqueueing more slices.

Slices are allowed to overlap on the GPU (no per-slice host sync). When `--max-inflight` is
reached, the oldest completed slice is drained (event sync + optional SMID memcpy + stream
destroy) to avoid unbounded resource growth.

This repeats until the full GEMM is complete (or `--max-slices` is hit as a safety cap).

It records:
- **where work runs** (sampled per-block SMIDs versus expected SM range per slice),
- **how fast mask switching happens** (`switch_us`, `set_plus_launch_us`, `switches_per_sec`),
- and emits CSV outputs suitable for plotting and later multi-process planning.

## Build

From `A100-partitioning/libsmctrl`:

`make libsmctrl_single_process_dynamic_mask_matmul`

## Run

From `A100-partitioning/libsmctrl`:

`../tests/mask_switch_rate_single_process_dynamic/libsmctrl_single_process_dynamic_mask_matmul --sm-ranges 6-9,10-25,26-35 --slice-rows 256 --max-slices 1000000 --max-inflight 4096 --warmup 5 --hold-iters 1 --sample-every 1 --m 2048 --n 2048 --k 2048 --csv-prefix single_process_dynamic_mask_matmul`

Useful knobs:
- `--sm-ranges`: SM range schedule, e.g. `6-9,10-25,26-35`; each slice uses one range.
- `--slice-rows`: how many output rows each GEMM slice computes (smaller slices => more streams/masks).
- `--max-inflight`: max concurrently queued slice streams before draining the oldest completed slice.
- `--max-slices`: safety cap on number of slices (should be larger than `ceil(M / slice_rows)`).
- `--hold-iters`: hold current mask for N slices before moving to next SM range.
- `--sample-every`: only sample SM residency every N slices.
- `--m --n --k`: GEMM shape.

## Outputs

Outputs are written in `tests/mask_switch_rate_single_process_dynamic`:
- `<prefix>_iterations.csv`
- `<prefix>_summary.csv`

## Plot

From `A100-partitioning/libsmctrl`:

`python3 ../tests/mask_switch_rate_single_process_dynamic/plot_single_process_dynamic_mask_matmul.py ../tests/mask_switch_rate_single_process_dynamic/single_process_dynamic_mask_matmul_iterations.csv ../tests/mask_switch_rate_single_process_dynamic/single_process_dynamic_mask_matmul_summary.csv`

This produces:
- `<prefix>_residency_timeline.png` — expected vs observed SM range (sampled slices); observed bands colored by `mask_id`
- `<prefix>_deploy_latency_us.png` — `switch_us`, `launch_us`, `set_plus_launch_us` per slice
- `<prefix>_latency_hists.png` — distributions of mask API vs full host enqueue window

## Metric Notes

After `--warmup` iterations, the binary runs one extra **prime** slice that matches the measured path:
`cudaMalloc` for the per-slice SMID buffer (same size as the first real slice), `cudaEventCreate` /
`cudaEventRecord`, sync, and teardown. The original warmup loop did not touch those APIs, so the
first row in the CSV could still show a large `set_plus_launch_us` even with a generous `--warmup`.

- `switch_us`: host-side time in `libsmctrl_set_stream_mask_ext(...)`.
- `launch_us`: host-side time from stream creation through GEMM launch enqueue (includes stream create + mask + enqueue).
- `set_plus_launch_us`: host-side time from before `cudaStreamCreate` through GEMM launch enqueue (does not include waiting for GPU completion).
- `inside_pct`/`outside_pct`: sampled block percentage running inside/outside the expected SM range.

Note: libsmctrl masks are **TPC-level**. SM ranges are mapped to TPC enable bits using a simple
`SM / (num_sms / num_tpcs)` heuristic, so strict SM-ID confinement may not always be possible.
