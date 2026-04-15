# Pure-Speed Mask+Deploy Benchmark

This benchmark measures only:
- how fast `libsmctrl_set_stream_mask_ext(...)` runs, and
- how fast a new kernel can be launched immediately afterwards.

No checker process is involved, and no per-iteration correctness validation is done. This is intentionally deployment-cadence only.

No CLI configuration is required. Constants are set at the top of `libsmctrl_dual_process_mask_check.cu`.

## Build

From `A100-partitioning/libsmctrl`:

`make libsmctrl_dual_process_mask_check`

## Run

From `A100-partitioning/libsmctrl`:

`../tests/mask_switch_rate_dual_process/libsmctrl_dual_process_mask_check`

## Outputs

Outputs are written in `tests/mask_switch_rate_dual_process`:
- `dual_process_mask_check_iterations.csv`
- `dual_process_mask_check_summary.csv`

## Plot

From `A100-partitioning/libsmctrl`:

`python3 ../tests/mask_switch_rate_dual_process/plot_dual_process_mask_check.py ../tests/mask_switch_rate_dual_process/dual_process_mask_check_iterations.csv ../tests/mask_switch_rate_dual_process/dual_process_mask_check_summary.csv`

This produces:
- `dual_process_mask_check_iterations_latency_us.png`
- `dual_process_mask_check_iterations_throughput_lps.png` (iteration vs `set_and_launch_us` cycle-style plot)
- `dual_process_mask_check_iterations_set_and_launch_hist.png`

## What each metric means

- `switch_us`: host-side duration of `libsmctrl_set_stream_mask_ext(...)`.
- `launch_us`: host-side duration of kernel launch enqueue call (`<<<...>>>` + launch error check).
- `set_and_launch_us`: host-side combined duration of setting mask then launching kernel.
- `launches_per_sec`: deployment throughput computed from the measured producer loop.
