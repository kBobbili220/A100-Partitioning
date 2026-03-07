#!/bin/bash
# Copyright 2025 Joshua Bakita
# Run the baseline, MPS, nvtaskset/libsmctrl, and MiG evaluation expriments from Bakita and Anderson (ECRTS 2025).
# Assumes that gpu-microbench, libsmctrl, cuda_scheduling_examiner, and nvdebug have been cloned and built, that nvdebug.ko has been loaded, and that the other repositories are available as gpu-microbench, libsmctrl, and cuda_scheduling_examiner_mirror from the current directory.
# (Use https://www.cs.unc.edu/~jbakita/rtas24-ae/clone_and_build_ecrts25.sh to set this up automatically.)
# What GPCs to give the task under test (A), and the three competing tasks (B-D), respectively
# The GPCs in GPCS_A should be mutually exclusive from those in GPCS_B, GCPS_C, or GPCS_D
# (defaults are configured for the NVIDIA A100 GPU)
# Configured for this A100: GPCs 0,1 have 7 TPCs; GPCs 2-6 have 8 TPCs each
GPCS_A=1,2,3,4
GPCS_B=0
GPCS_C=5
GPCS_D=6

# Percentage of GPU to give the task under test (A), and the three competing tasks (B-D), respectively
# The total of these values must equal 100
# (defaults are configured for the NVIDIA A100 GPU)
# Calculated based on TPC counts: A=31 TPCs, B=7 TPCs, C=8 TPCs, D=8 TPCs (total=54 TPCs)
PCT_A=57
PCT_B=13
PCT_C=15
PCT_D=15

# Configuration for RTX 2080 Ti
#GPCS_A=0,1,2
#GPCS_B=3
#GPCS_C=4
#GPCS_D=5
#PCT_A=50
#PCT_B=17
#PCT_C=17
#PCT_D=16

# How many samples to gather within each experiment class (set to zero to disable)
# (defaults are the sample counts used in the paper)
STARTUP_OH_SAMPLES=1000
LAUNCH_OH_SAMPLES=1000000
ENFORCEMENT_SAMPLES=1000
GRANULARITY_SAMPLES=10

# Print sample configuration
echo -e "┌────────────── evaluate_ecrts25.sh ──────────────"
echo -e "│ Enabled Experiments:"
if [ $STARTUP_OH_SAMPLES -gt 0 ]; then
  echo -e "│ + Startup Overhead. Sample Count: $STARTUP_OH_SAMPLES"
fi
if [ $LAUNCH_OH_SAMPLES -gt 0 ]; then
  echo -e "│ + Launch Overhead. Sample Count: $LAUNCH_OH_SAMPLES"
fi
if [ $ENFORCEMENT_SAMPLES -gt 0 ]; then
  echo -e "│ + Partition Enforcement. Sample Count: $ENFORCEMENT_SAMPLES"
fi
if [ $GRANULARITY_SAMPLES -gt 0 ]; then
  echo -e "│ + Partition Granularity. Sample Count: $GRANULARITY_SAMPLES"
fi
echo -e "│ Disabled Experiments:"
if (( $STARTUP_OH_SAMPLES + $LAUNCH_OH_SAMPLES + ENFORCEMENT_SAMPLES + GRANULARITY_SAMPLES != 0 )); then
  if [ $STARTUP_OH_SAMPLES -eq 0 ]; then
    echo -e "│ + Startup Overhead"
  fi
  if [ $LAUNCH_OH_SAMPLES -eq 0 ]; then
    echo -e "│ + Launch Overhead"
  fi
  if [ $ENFORCEMENT_SAMPLES -eq 0 ]; then
    echo -e "│ + Partition Enforcement"
  fi
  if [ $GRANULARITY_SAMPLES -eq 0 ]; then
    echo -e "│ + Partition Granularity"
  fi
else
  echo -e "│  [None]"
fi
echo -e "└─────────────────────────────────────────────────"

echo -e "\e[4m\e[1m***** Verifying Configuration *****\e[0m"

# Verify that the requisite benchmarks are available
if [ ! -d "cuda_scheduling_examiner_mirror" -o ! -d "libsmctrl" -o ! -d "gpu-microbench" ]; then
  echo "Run the setup instructions first! The required repositiories are not available."
  exit 1
fi

# Terminate MPS if it was left running by an earlier, interrupted run
echo "quit" | nvidia-cuda-mps-control 2> /dev/null

# Set the GPU ID to use (default: 0, can be overridden via environment)
GPU_ID="${GPU_ID:-0}"

# Check that the GPU is idle
if [ $(nvidia-smi -i $GPU_ID --query-compute-apps=pid --format=csv | wc -l) -ne 1 ]; then
  echo "GPU$GPU_ID does not appear to be idle. Please see nvidia-smi and terminate any applications it lists as using GPU$GPU_ID."
  exit 1
fi

# Verify that nvdebug is still loaded
if [ ! -e /proc/gpu$GPU_ID ]; then
  sudo insmod nvdebug/nvdebug.ko
fi

# Run everything on the first GPU by PCIe ID
# (changing this may not be sufficient to run on another GPU; `libsmctrl_test_gpc_info` and `test_granularity.py` assume use of GPU 0)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Mitigate the MPS issue with not creating enough channels
export CUDA_DEVICE_MAX_CONNECTIONS=8

# Tell the loader where to find libsmctrl.so and the fake libcuda.so.1
# NOTE: This is cleared later for non-libsmctrl-wrapper startup overhead experiments to prevent automatic loading of the fake libcuda.so.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/libsmctrl
export PYTHONPATH=$PYTHONPATH:$(pwd)/libsmctrl

# Not strictly necessary, but makes MPS stop incessently warning about it not being able to log to /var/log/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp

# Use libsmctrl_test_gpc_info to derive the SM masks that correspond to the above GPC settings
gpcs_to_mask() {
  IFS=, read -a GPCS <<< $1
  MASK_X=0
  for num in ${GPCS[@]}; do
   idx=$(( $num + 2 ))
   mask=$(./libsmctrl/libsmctrl_test_gpc_info "$GPU_ID" | head -$idx | tail -1 | cut -d " " -f 10)
   MASK_X=$(( $MASK_X | $mask ))
  done
  printf "0x%016llx" $MASK_X
}
# Must add inversion prefix to convert the disable mask to an enable mask (only for non-MiG tests)
if [ $# -eq 0 -o "$1" != "mig" ]; then
  MASK_A_ENABLE=$(gpcs_to_mask $GPCS_A)
  MASK_A=~$MASK_A_ENABLE
  MASK_B=~$(gpcs_to_mask $GPCS_B)
  MASK_C=~$(gpcs_to_mask $GPCS_C)
  MASK_D=~$(gpcs_to_mask $GPCS_D)
fi

# Helper function for cleaning up the output from cuda_scheduling_examiner
# Takes one argument: the file name (without .json)
strip_and_copy() {
  cat ./results/$1.json | jq "[.times[].execute_times | select(. != null) | (.[1] - .[0]) * 1000]" > ../$1_stripped.json
}

eval_baseline() {
## Baseline (overhead and enforcement)
echo -e "\e[4m\e[1m***** Evaluating Baseline (no partitioning) *****\e[0m"
cd gpu-microbench
if [ $STARTUP_OH_SAMPLES -gt 0 ]; then
  # Run benchmark once in a "warmup" round to pull relevant binaries into the page cache
  LD_LIBRARY_PATH="" ./measure_startup_oh > /dev/null 2>&1
  for (( i=0; i<$STARTUP_OH_SAMPLES; i+=1 )); do
    LD_LIBRARY_PATH="" ./measure_startup_oh 2>&1 | cut -d "=" -s -f 2 | cut -d " " -f 2 | tr "\n" " " | sed 's/$/r - p/' | dc >> ../startup_oh_baseline.log;
  done
fi
if [ $LAUNCH_OH_SAMPLES -gt 0 ]; then
  taskset -c 5 ./measure_launch_oh $LAUNCH_OH_SAMPLES >> ../launch_oh_baseline.log
fi
cd ..
cd cuda_scheduling_examiner_mirror
if [ $ENFORCEMENT_SAMPLES -gt 0 ]; then
  # Update the benchmark to use the requested sample count
  jq ".benchmarks[0].max_iterations = $ENFORCEMENT_SAMPLES" configs/ecrts25_isol_baseline.json > configs/ecrts25_isol_baseline_custom.json
  jq ".benchmarks[0].max_iterations = $ENFORCEMENT_SAMPLES" configs/ecrts25_isol_none_mb.json > configs/ecrts25_isol_none_mb_custom.json
  jq ".benchmarks[0].max_iterations = $ENFORCEMENT_SAMPLES" configs/ecrts25_isol_none_rw.json > configs/ecrts25_isol_none_rw_custom.json
  ./bin/runner configs/ecrts25_isol_baseline_custom.json
  ./bin/runner configs/ecrts25_isol_none_mb_custom.json
  ./bin/runner configs/ecrts25_isol_none_rw_custom.json
  strip_and_copy ecrts25_isol_baseline
  strip_and_copy ecrts25_isol_none_mb
  strip_and_copy ecrts25_isol_none_rw
fi
cd ..
}

eval_mps() {
## MPS (overhead, granularity, and enforcement)
echo -e "\e[4m\e[1m***** Evaluating MPS *****\e[0m"
# Ensure CUDA_VISIBLE_DEVICES is set before starting MPS (MPS reads it at startup)
export CUDA_VISIBLE_DEVICES=$GPU_ID
# Start MPS with the selected GPU
nvidia-cuda-mps-control -d
# After MPS starts, unset CUDA_VISIBLE_DEVICES - MPS manages device access now
unset CUDA_VISIBLE_DEVICES
nvidia-smi -L
./gpu-microbench/constant_cycles_kernel 1 # To warm up MPS

cd gpu-microbench
if [ $STARTUP_OH_SAMPLES -gt 0 ]; then
  # Run benchmark once in a "warmup" round to pull relevant binaries into the page cache
  LD_LIBRARY_PATH="" CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$PCT_A ./measure_startup_oh > /dev/null 2>&1
  for (( i=0; i<$STARTUP_OH_SAMPLES; i+=1 )); do
    LD_LIBRARY_PATH="" CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$PCT_A ./measure_startup_oh 2>&1 | cut -d "=" -s -f 2 | cut -d " " -f 2 | tr "\n" " " | sed 's/$/r - p/' | dc >> ../startup_oh_mps.log;
  done
fi
if [ $LAUNCH_OH_SAMPLES -gt 0 ]; then
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$PCT_A taskset -c 5 ./measure_launch_oh $LAUNCH_OH_SAMPLES >> ../launch_oh_mps.log
fi
cd ..
cd cuda_scheduling_examiner_mirror
if [ $ENFORCEMENT_SAMPLES -gt 0 ]; then
  # Update the benchmark to use the requested sample count and partitioning for this platform
  jq ".benchmarks[0].max_iterations = $ENFORCEMENT_SAMPLES | .benchmarks[0].mps_thread_percentage = $PCT_A | .benchmarks[1].mps_thread_percentage = $PCT_B | .benchmarks[2].mps_thread_percentage = $PCT_C | .benchmarks[3].mps_thread_percentage = $PCT_D" configs/ecrts25_isol_mps_mb.json > configs/ecrts25_isol_mps_mb_custom.json
  jq ".benchmarks[0].max_iterations = $ENFORCEMENT_SAMPLES | .benchmarks[0].mps_thread_percentage = $PCT_A | .benchmarks[1].mps_thread_percentage = $PCT_B | .benchmarks[2].mps_thread_percentage = $PCT_C | .benchmarks[3].mps_thread_percentage = $PCT_D" configs/ecrts25_isol_mps_rw.json > configs/ecrts25_isol_mps_rw_custom.json
  ./bin/runner configs/ecrts25_isol_mps_mb_custom.json
  ./bin/runner configs/ecrts25_isol_mps_rw_custom.json
  strip_and_copy ecrts25_isol_mps_mb
  strip_and_copy ecrts25_isol_mps_rw
fi
if [ $GRANULARITY_SAMPLES -gt 0 ]; then
  python3 ./scripts/test_granularity.py -i $GRANULARITY_SAMPLES # Auto-detects and adjusts for platform
fi
cd ..

echo "quit" | nvidia-cuda-mps-control
# Restore CUDA_VISIBLE_DEVICES after MPS stops (for non-MPS experiments)
export CUDA_VISIBLE_DEVICES=$GPU_ID
}

eval_libsmctrl() {
## libsmctrl/nvtaskset ("nvsplit") (overhead and enforcement) (granularity included in above)
echo -e "\e[4m\e[1m***** Evaluating libsmctrl/nvtaskset *****\e[0m"
nvidia-cuda-mps-control -d
unset CUDA_VISIBLE_DEVICES
./gpu-microbench/constant_cycles_kernel 1 # To warm up MPS

cd gpu-microbench
if [ $STARTUP_OH_SAMPLES -gt 0 ]; then
  # Run each benchmark configuration once in a "warmup" round to pull relevant binaries into the page cache
  LD_LIBRARY_PATH="" ./measure_startup_oh -e LD_PRELOAD=../libsmctrl/libsmctrl.so > /dev/null 2>&1
  ./measure_startup_oh -e LIBSMCTRL_MASK=$MASK_A > /dev/null 2>&1
  ./measure_startup_oh ../libsmctrl/nvtaskset $MASK_A_ENABLE ./measure_launch_oh 1 > /dev/null 2>&1
  ./measure_startup_oh ../libsmctrl/nvtaskset --gpc-list $GPCS_A ./measure_launch_oh 1 > /dev/null 2>&1
  for (( i=0; i<$STARTUP_OH_SAMPLES; i+=1 )); do
    # Just cost of loading the library
    LD_LIBRARY_PATH="" ./measure_startup_oh -e LD_PRELOAD=../libsmctrl/libsmctrl.so 2>&1 | cut -d "=" -s -f 2 | cut -d " " -f 2 | tr "\n" " " | sed 's/$/r - p/' | dc >> ../startup_oh_libsmctrl.log;
    # Cost of libsmctrl-wrapper (assumes fake libsmctrl.so.1 is on LD_LIBRARY_PATH)
    ./measure_startup_oh -e LIBSMCTRL_MASK=$MASK_A 2>&1 | cut -d "=" -s -f 2 | cut -d " " -f 2 | tr "\n" " " | sed 's/$/r - p/' | dc >> ../startup_oh_libsmctrl-wrapper.log;
    # nvtaskset without GPC lookup
    ./measure_startup_oh ../libsmctrl/nvtaskset $MASK_A_ENABLE ./measure_launch_oh 1 2>&1 | cut -d "=" -s -f 2 | cut -d " " -f 2 | tr "\n" " " | sed 's/$/r - p/' | dc >> ../startup_oh_nvtaskset.log;
    # nvtaskset with GPC lookup
    ./measure_startup_oh ../libsmctrl/nvtaskset --gpc-list $GPCS_A ./measure_launch_oh 1 2>&1 | cut -d "=" -s -f 2 | cut -d " " -f 2 | tr "\n" " " | sed 's/$/r - p/' | dc >> ../startup_oh_nvtaskset-gpc.log;
  done
fi
if [ $LAUNCH_OH_SAMPLES -gt 0 ]; then
  taskset -c 5 ../libsmctrl/nvtaskset --gpc-list $GPCS_A ./measure_launch_oh $LAUNCH_OH_SAMPLES >> ../launch_oh_libsmctrl.log
fi
cd ..
cd cuda_scheduling_examiner_mirror
if [ $ENFORCEMENT_SAMPLES -gt 0 ]; then
  # Update the benchmark to use the requested sample count and partitioning for this platform
  # (we cannot directly use nvtaskset, since ./bin/runner spawns off several benchmarks that must each go in its own partition)
  jq ".benchmarks[0].max_iterations = $ENFORCEMENT_SAMPLES | .benchmarks[0].sm_mask = \"$MASK_A\" | .benchmarks[1].sm_mask = \"$MASK_B\" | .benchmarks[2].sm_mask = \"$MASK_C\" | .benchmarks[3].sm_mask = \"$MASK_D\"" configs/ecrts25_isol_libsmctrl_mb.json > configs/ecrts25_isol_libsmctrl_mb_custom.json
  jq ".benchmarks[0].max_iterations = $ENFORCEMENT_SAMPLES | .benchmarks[0].sm_mask = \"$MASK_A\" | .benchmarks[1].sm_mask = \"$MASK_B\" | .benchmarks[2].sm_mask = \"$MASK_C\" | .benchmarks[3].sm_mask = \"$MASK_D\"" configs/ecrts25_isol_libsmctrl_rw.json > configs/ecrts25_isol_libsmctrl_rw_custom.json
  ./bin/runner configs/ecrts25_isol_libsmctrl_mb_custom.json
  ./bin/runner configs/ecrts25_isol_libsmctrl_rw_custom.json
  strip_and_copy ecrts25_isol_libsmctrl_mb
  strip_and_copy ecrts25_isol_libsmctrl_rw
fi
cd ..

echo "quit" | nvidia-cuda-mps-control
}

eval_mig() {
echo -e "\e[4m\e[1m***** Evaluating MiG *****\e[0m"
# Reset MiG (if needed)
sudo nvidia-smi mig -dci 2> /dev/null
sudo nvidia-smi mig -dgi 2> /dev/null
# Set up the 57% partition for the bencmark under test and get its UUID
sudo nvidia-smi mig -cgi 4g.20gb -C # Primary
MIG_PRIMARY_UUID=$(nvidia-smi -L | tail -1 | cut -d ":" -f 3 | tr -d " " | tr -d ")")
# Run overhead experiments
cd gpu-microbench
if [ $STARTUP_OH_SAMPLES -gt 0 ]; then
  # Run benchmark once in a "warmup" round to pull relevant binaries into the page cache
  LD_LIBRARY_PATH="" ./measure_startup_oh > /dev/null 2>&1
  for (( i=0; i<$STARTUP_OH_SAMPLES; i+=1 )); do
    LD_LIBRARY_PATH="" ./measure_startup_oh 2>&1 | cut -d "=" -s -f 2 | cut -d " " -f 2 | tr "\n" " " | sed 's/$/r - p/' | dc >> ../startup_oh_mig.log;
  done
fi
if [ $LAUNCH_OH_SAMPLES -gt 0 ]; then
  taskset -c 5 ./measure_launch_oh $LAUNCH_OH_SAMPLES >> ../launch_oh_mig.log
fi
cd ..
# Set up the 43% partition for competing work, get its UUID, and initialize MPS on it
sudo nvidia-smi mig -cgi 3g.20gb -C
MIG_SECONDARY_UUID=$(nvidia-smi -L | tail -1 | cut -d ":" -f 3 | tr -d " " | tr -d ")")
if [ "$MIG_PRIMARY_UUID" = "$MIG_SECONDARY_UUID" ]; then
  MIG_SECONDARY_UUID=$(nvidia-smi -L | tail -2 | head -1 | cut -d ":" -f 3 | tr -d " " | tr -d ")")
fi
CUDA_VISIBLE_DEVICES=$MIG_SECONDARY_UUID nvidia-cuda-mps-control -d
CUDA_VISIBLE_DEVICES=$MIG_SECONDARY_UUID ./gpu-microbench/constant_cycles_kernel 1 # To warm up MPS
# Run partition enforcement experiments
cd cuda_scheduling_examiner_mirror
if [ $ENFORCEMENT_SAMPLES -gt 0 ]; then
  # Update the benchmark to use the requested sample count
  jq ".benchmarks[0].max_iterations = $ENFORCEMENT_SAMPLES" configs/ecrts25_isol_mig_rw_2.json > configs/ecrts25_isol_mig_rw_2_custom.json
  jq ".benchmarks[0].max_iterations = $ENFORCEMENT_SAMPLES" configs/ecrts25_isol_mig_mb_2.json > configs/ecrts25_isol_mig_mb_2_custom.json
  CUDA_VISIBLE_DEVICES=$MIG_SECONDARY_UUID ./bin/runner configs/ecrts25_isol_mig_rw_1.json & # Competing work
  sleep 20 # Let competitor start
  CUDA_VISIBLE_DEVICES=$MIG_PRIMARY_UUID CUDA_MPS_PIPE_DIRECTORY=/dev/null ./bin/runner configs/ecrts25_isol_mig_rw_2_custom.json
  killall runner # This will also crash MPS
  echo "quit" | nvidia-cuda-mps-control # Reset MPS
  CUDA_VISIBLE_DEVICES=$MIG_SECONDARY_UUID nvidia-cuda-mps-control -d
  CUDA_VISIBLE_DEVICES=$MIG_SECONDARY_UUID ../gpu-microbench/constant_cycles_kernel 1 # To warm up MPS again
  CUDA_VISIBLE_DEVICES=$MIG_SECONDARY_UUID ./bin/runner configs/ecrts25_isol_mig_mb_1.json & # Competing work
  sleep 20 # Let competitor start
  CUDA_VISIBLE_DEVICES=$MIG_PRIMARY_UUID CUDA_MPS_PIPE_DIRECTORY=/dev/null ./bin/runner configs/ecrts25_isol_mig_mb_2_custom.json
  killall runner
  strip_and_copy ecrts25_isol_mig_mb
  strip_and_copy ecrts25_isol_mig_rw
fi
# Reset configuration for test_granularity.py
echo "quit" | nvidia-cuda-mps-control
if [ $GRANULARITY_SAMPLES -gt 0 ]; then
  sudo nvidia-smi mig -dci
  sudo nvidia-smi mig -dgi
  # Run granularity experiments
  # The TPC count has to be manually specified, since test_granularity.py cannot determine the hardware TPC count while MiG is in use
  python3 ./scripts/test_granularity.py --mig --device 0 --tpc_count 54 --iterations $GRANULARITY_SAMPLES # Auto-detects and adjusts for platform
  sudo nvidia-smi mig -dci
  sudo nvidia-smi mig -dgi
fi
cd ..
}

# Optional argument: which mechanism to evaluate
if [ $# -gt 0 ]; then
  if [ $1 = "baseline" ]; then
    eval_baseline
  elif [ $1 = "mps" ]; then
    eval_mps
  elif [ $1 = "libsmctrl" ]; then
    eval_libsmctrl
  elif [ $1 = "mig" ]; then
    eval_mig
  else
    echo "Unrecognized argument: $1"
    echo "Usage: $0 <baseline, mps, libsmctrl, or mig>"
  fi
else
  # Default: evaluate everything but MiG
  eval_baseline
  eval_mps
  eval_libsmctrl
fi

echo -e "\e[4m\e[1m***** Complete *****\e[0m"
