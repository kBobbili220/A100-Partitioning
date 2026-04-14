# Dynamic Switch Timeline Demo

Build from `libsmctrl/`:

`make libsmctrl_dynamic_switch_demo`

Run:

`../tests/dynamic_switch/libsmctrl_dynamic_switch_demo <next|global|stream>`

Artifacts in this directory:
- `dynamic_switch_*.csv`
- `dynamic_switch_*_durations.png`
- `dynamic_switch_*_heatmap.png`

Plot script:

`python3 ../tests/dynamic_switch/plot_dynamic_switch_timeline.py <timeline.csv>`
