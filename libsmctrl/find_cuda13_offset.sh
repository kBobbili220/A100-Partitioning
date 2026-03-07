#!/bin/bash
# find_cuda13_offset.sh
# Finds the correct stream struct TPC mask offset for CUDA 13.0 by using the
# MASK_OFF environment variable probe mechanism built into libsmctrl.
#
# HOW IT WORKS:
# libsmctrl_set_stream_mask_ext() contains this code (libsmctrl.c ~line 431):
#
#   char* mask_off_str = getenv("MASK_OFF");
#   if (mask_off_str) {
#       int off = atoi(mask_off_str);
#       hw_mask_v2 = (void*)(stream_struct_base + CU_12_2_MASK_OFF + off);
#   }
#
# When MASK_OFF is set, it BYPASSES the version switch entirely and writes the
# stream_sm_mask_v2 struct at: stream_base + 0x4e4 + MASK_OFF
#
# The functional test (libsmctrl_test_stream_mask) launches 142 blocks and
# checks that they all ran on exactly 1 TPC's SMs. It passes IFF the mask worked.
#
# SEARCH SPACE:
# Known offsets for reference (absolute values):
#   CUDA 12.2: 0x4e4 (1252) → MASK_OFF = 0
#   CUDA 12.3: 0x49c (1180) → MASK_OFF = -72
#   CUDA 12.4: 0x4ac (1196) → MASK_OFF = -56
#   CUDA 12.5/12.6: 0x4ec (1260) → MASK_OFF = 8
#   CUDA 12.7/12.8: 0x4fc (1276) → MASK_OFF = 24  ← user tried this, FAILED
#
# We scan from -400 to +400 in steps of 4. Absolute offset = 0x4e4 + MASK_OFF
# Range covers 0x354 (852) through 0x674 (1652).

BASE_OFF=0x4e4   # CU_12_2_MASK_OFF in decimal
BASE_DEC=1252
STEP=4
MIN_OFF=-400
MAX_OFF=400
TEST_TIMEOUT=10  # seconds per test

# Allow command-line overrides
if [ "$1" != "" ]; then
    MIN_OFF=$1
fi
if [ "$2" != "" ]; then
    MAX_OFF=$2
fi
if [ "$3" != "" ]; then
    TEST_TIMEOUT=$3
fi

PASSING=()
TESTED=0

echo "================================================================="
echo " libsmctrl CUDA 13.0 stream mask offset probe"
echo " Scanning MASK_OFF from $MIN_OFF to $MAX_OFF in steps of $STEP"
echo " Test timeout: ${TEST_TIMEOUT}s per offset"
echo " Absolute offset = 0x4e4 + MASK_OFF = $BASE_DEC + MASK_OFF"
echo "================================================================="
echo ""

# First, verify the test binary exists
if [ ! -x ./libsmctrl_test_stream_mask ]; then
    echo "ERROR: ./libsmctrl_test_stream_mask not found or not executable."
    echo "Build it with: make libsmctrl_test_stream_mask"
    exit 1
fi

# Verify CUDA 13.0 abort is bypassed by MASK_OFF=0
echo "[Sanity check] Testing MASK_OFF=0 (absolute offset 0x4e4)..."
result=$(MASK_OFF=0 ./libsmctrl_test_stream_mask 2>&1)
if echo "$result" | grep -q "unimplemented\|abort\|Aborted"; then
    echo "ERROR: Even with MASK_OFF set, the binary aborts. Something is wrong."
    echo "Output was: $result"
    exit 1
fi
echo "[Sanity check] MASK_OFF env var works (no abort). Starting scan..."
echo ""

# Main scan loop
for (( off=MIN_OFF; off<=MAX_OFF; off+=STEP )); do
    abs=$(( BASE_DEC + off ))
    abs_hex=$(printf "0x%x" $abs)

    # Run test with timeout to prevent hanging
    result=$(timeout ${TEST_TIMEOUT} bash -c "MASK_OFF=$off ./libsmctrl_test_stream_mask 2>&1")
    test_exit=$?
    TESTED=$(( TESTED + 1 ))

    # Check for timeout (exit code 124 from timeout command)
    if [ $test_exit -eq 124 ]; then
        echo "[TIMEOUT] MASK_OFF=$off ($abs_hex) - test hung after ${TEST_TIMEOUT}s"
    elif [ $test_exit -ne 0 ]; then
        # Test crashed or returned error (only print every 20 to reduce noise)
        if (( TESTED % 20 == 0 )); then
            echo "  [progress] tested $TESTED offsets, last: MASK_OFF=$off ($abs_hex) exit=$test_exit"
        fi
    elif echo "$result" | grep -q "Test passed"; then
        echo "*** PASS: MASK_OFF=$off  →  absolute offset $abs_hex ($abs) ***"
        PASSING+=($off)
    else
        # Test ran but didn't pass (only print every 20 to reduce noise)
        if (( TESTED % 20 == 0 )); then
            echo "  [progress] tested $TESTED offsets, last: MASK_OFF=$off ($abs_hex)"
        fi
    fi
done

echo ""
echo "================================================================="
echo " SCAN COMPLETE: tested $TESTED offsets"
echo "================================================================="

if [ ${#PASSING[@]} -eq 0 ]; then
    echo ""
    echo "RESULT: No offset passed the stream mask test."
    echo "Possible causes:"
    echo "  1. The stream struct offset changed significantly in CUDA 13.0"
    echo "     Try widening the search: edit MIN_OFF/MAX_OFF in this script or use:"
    echo "     ./find_cuda13_offset.sh -500 500 10"
    echo "  2. MPS is not running (stream masking requires MPS)"
    echo "  3. The test binary was built against an old libsmctrl"
    echo "  4. One of the offsets is causing the test to hang (see TIMEOUT messages above)"
    echo ""
    echo "Checking MPS status:"
    echo "quit" | nvidia-cuda-mps-control 2>/dev/null && echo "MPS was running, now stopped" || echo "MPS was not running"
    echo ""
    echo "NOTE: The libsmctrl_test_stream_mask test does NOT require MPS —"
    echo "stream masks work without it (only enforcement co-running needs MPS)."
else
    echo ""
    echo "RESULT: The following MASK_OFF values passed the stream mask test:"
    for off in "${PASSING[@]}"; do
        abs=$(( BASE_DEC + off ))
        abs_hex=$(printf "0x%x" $abs)
        echo "  MASK_OFF=$off  →  #define CU_13_0_MASK_OFF $abs_hex"
    done
    echo ""
    echo "NEXT STEP: Add the correct case to libsmctrl.c:"
    echo "  In libsmctrl_set_stream_mask_ext(), inside the #if __x86_64__ block:"
    echo ""
    best_off="${PASSING[0]}"
    best_abs=$(( BASE_DEC + best_off ))
    best_hex=$(printf "0x%x" $best_abs)
    cat << EOF
  // Add near line 385 (after case 12080):
  case 13000:
      hw_mask_v2 = (void*)(stream_struct_base + $best_hex);
      break;
  // And add this define near line 251:
  #define CU_13_0_MASK_OFF $best_hex
  // 13.0 tested on <your driver version>
EOF
fi
echo "================================================================="
