# Mask Switch Rate Benchmark

Build from `libsmctrl/`:

`make libsmctrl_mask_switch_rate_demo`

Run switch-rate benchmark (single process, single stream, round-robin one TPC at a time):

`../tests/mask_switch_rate/libsmctrl_mask_switch_rate_demo --iters 400 --warmup 20 --csv-prefix mask_switch_rate`

Optionally select only a subset of TPCs:

`../tests/mask_switch_rate/libsmctrl_mask_switch_rate_demo --tpcs 0,1,4-7 --iters 400 --warmup 20`

What the probe does:
- Per iteration, it sets a stream mask for one requested TPC, launches a minimal 1-block/1-thread probe kernel, synchronizes, and copies one SMID sample back to host.
- It infers one observed TPC from that SMID and records correctness (`inside/outside`, `dominant_tpc`, `mismatch`) plus timing (`switch_us`, `cycle_us`).
- This benchmark is intentionally focused on fast mask-set + kernel-deploy cadence (no large matrix math and no block-count parameter).

Outputs are written to `tests/mask_switch_rate`:
- `<prefix>_iterations.csv`
- `<prefix>_summary.csv`

Plot:

`python3 ../tests/mask_switch_rate/plot_mask_switch_rate.py ../tests/mask_switch_rate/mask_switch_rate_iterations.csv ../tests/mask_switch_rate/mask_switch_rate_summary.csv`
