// Time-resolved dynamic switching demo:
// - Launches a long-running compute kernel in fixed "windows"
// - Reconfigures which TPC is enabled between windows
// - Samples block->SM placement to build per-window SM histograms
// - Prints an ASCII timeline in terminal for quick inspection
// - Exports full per-SM counts to CSV for plotting

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "libsmctrl.h"

static void die(const char* msg) {
  fprintf(stderr, "fatal: %s\n", msg);
  std::exit(1);
}

static void cuda_check(cudaError_t e, const char* expr) {
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s (%d) at %s\n", cudaGetErrorString(e), (int)e, expr);
    std::exit(1);
  }
}

static uint64_t now_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

// GEMM-like probe kernel.
// Purpose:
// 1) stay resident long enough to make window timing meaningful,
// 2) record one SMID per block to understand scheduling behavior,
// 3) write a tiny output value so work is not optimized away.
__global__ void gemm_like_probe_kernel(float* out, uint16_t* smid_per_block, int repeats) {
  __shared__ float A[16][16];
  __shared__ float B[16][16];
  __shared__ float C[16][16];

  // 256 threads arranged logically as a 16x16 tile.
  int tx = threadIdx.x & 15;
  int ty = threadIdx.x >> 4;  // 0..15 when blockDim.x = 256

  // Deterministic initialization of shared tiles.
  if (tx < 16 && ty < 16) {
    A[ty][tx] = (float)((tx + ty + blockIdx.x) & 7) + 1.0f;
    B[ty][tx] = (float)((tx * 3 + ty + 1) & 7) + 1.0f;
    C[ty][tx] = 0.0f;
  }
  __syncthreads();

  int smid = 0;
  asm("mov.u32 %0, %%smid;" : "=r"(smid));
  // One SMID sample per block (thread 0 only) keeps global writes cheap.
  if (threadIdx.x == 0) {
    smid_per_block[blockIdx.x] = (uint16_t)smid;
  }

  // Repeated tiny GEMM inner loop to create controllable compute duration.
  float acc = (float)(tx + 1) * 0.001f;
  for (int r = 0; r < repeats; r++) {
    #pragma unroll
    for (int k = 0; k < 16; k++) {
      acc += A[ty][k] * B[k][tx];
      acc = acc * 1.000001f + 0.0001f;
    }
  }

  if (tx < 16 && ty < 16) {
    C[ty][tx] = acc;
  }
  __syncthreads();

  // One output per block is enough to keep side effects visible.
  if (threadIdx.x == 0) {
    out[blockIdx.x] = C[0][0];
  }
}

// Collected metrics for one time window of execution.
struct WindowSummary {
  int win_idx;
  int phase_idx;
  int enabled_tpc;
  uint64_t start_ns;
  uint64_t end_ns;
  std::vector<int> sm_hist;
};

static void apply_mask_for_window(const std::string& method,
                                  cudaStream_t stream,
                                  int enabled_tpc) {
  // libsmctrl uses a "disable mask": bit=1 means disabled.
  // Start from all-disabled then flip bits so only one TPC stays enabled.
  uint128_t disable_mask = (uint128_t)1;
  disable_mask <<= enabled_tpc;
  disable_mask = ~disable_mask;  // enable exactly this TPC

  // Dispatch to the selected masking strategy.
  if (method == "next") {
    libsmctrl_set_next_mask((uint64_t)disable_mask);
  } else if (method == "global") {
    libsmctrl_set_global_mask((uint64_t)disable_mask);
  } else if (method == "stream") {
    libsmctrl_set_stream_mask_ext((void*)stream, disable_mask);
  } else {
    die("method must be one of: next | global | stream");
  }
}

static WindowSummary run_window(int win_idx,
                                int phase_idx,
                                int enabled_tpc,
                                const std::string& method,
                                cudaStream_t stream,
                                int blocks,
                                int repeats,
                                int num_sms,
                                float* d_out,
                                uint16_t* d_smids) {
  // 1) Program the scheduler mask for this window.
  apply_mask_for_window(method, stream, enabled_tpc);

  // 2) Run one kernel instance and time the full stream-complete window.
  uint64_t t0 = now_ns();
  gemm_like_probe_kernel<<<blocks, 256, 0, stream>>>(d_out, d_smids, repeats);
  cuda_check(cudaGetLastError(), "gemm_like_probe_kernel launch");
  cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  uint64_t t1 = now_ns();

  // 3) Pull SMID samples back and convert to histogram (blocks per SM).
  std::vector<uint16_t> h_smids(blocks);
  cuda_check(cudaMemcpy(h_smids.data(), d_smids, sizeof(uint16_t) * blocks, cudaMemcpyDeviceToHost),
             "cudaMemcpy smid D2H");

  WindowSummary s;
  s.win_idx = win_idx;
  s.phase_idx = phase_idx;
  s.enabled_tpc = enabled_tpc;
  s.start_ns = t0;
  s.end_ns = t1;
  s.sm_hist.assign(num_sms, 0);
  for (int i = 0; i < blocks; i++) {
    int sm = (int)h_smids[i];
    if (sm >= 0 && sm < num_sms) s.sm_hist[sm]++;
  }
  return s;
}

static void print_ascii_timeline_row(const WindowSummary& s) {
  // Single-line compact visualization:
  // each character column corresponds to one SM index.
  printf("window=%02d phase=%02d tpc=%d dur=%7.2f ms |",
         s.win_idx, s.phase_idx, s.enabled_tpc,
         (double)(s.end_ns - s.start_ns) / 1.0e6);
  for (size_t sm = 0; sm < s.sm_hist.size(); sm++) {
    // Visual intensity for this SM in this window.
    char c = '.';
    if (s.sm_hist[sm] >= 16) c = '#';
    else if (s.sm_hist[sm] >= 4) c = '*';
    else if (s.sm_hist[sm] > 0) c = '+';
    putchar(c);
  }
  printf("|\n");
}

static void write_csv(const std::vector<WindowSummary>& rows, const char* csv_path) {
  // Wide-to-long export: one CSV row per (window, smid) pair.
  FILE* f = fopen(csv_path, "w");
  if (!f) die("unable to open CSV output path");
  fprintf(f, "window,phase,enabled_tpc,start_ns,end_ns,duration_ms,smid,blocks\n");
  for (const auto& r : rows) {
    double dur_ms = (double)(r.end_ns - r.start_ns) / 1.0e6;
    for (size_t sm = 0; sm < r.sm_hist.size(); sm++) {
      fprintf(f, "%d,%d,%d,%llu,%llu,%.6f,%zu,%d\n",
              r.win_idx, r.phase_idx, r.enabled_tpc,
              (unsigned long long)r.start_ns,
              (unsigned long long)r.end_ns,
              dur_ms,
              sm, r.sm_hist[sm]);
    }
  }
  fclose(f);
}

int main(int argc, char** argv) {
  // CLI allows quick experiments without recompiling.
  if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
    fprintf(stderr,
            "Usage: %s <method> [windows=24] [windows_per_phase=4] [tpc_a=0] [tpc_b=1] "
            "[blocks=4096] [repeats=4096] [csv=dynamic_switch_timeline.csv]\n"
            "method: next | global | stream\n",
            argv[0]);
    return 1;
  }

  std::string method = argv[1];
  int windows = (argc >= 3) ? std::atoi(argv[2]) : 24;
  int windows_per_phase = (argc >= 4) ? std::atoi(argv[3]) : 4;
  int tpc_a = (argc >= 5) ? std::atoi(argv[4]) : 0;
  int tpc_b = (argc >= 6) ? std::atoi(argv[5]) : 1;
  int blocks = (argc >= 7) ? std::atoi(argv[6]) : 4096;
  int repeats = (argc >= 8) ? std::atoi(argv[7]) : 4096;
  const char* csv_path = (argc >= 9) ? argv[8] : "dynamic_switch_timeline.csv";

  // Query GPU topology once; it is used for validation and readable output.
  int num_sms = 0;
  cuda_check(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0),
             "cudaDeviceGetAttribute(MP count)");
  uint32_t num_tpcs = 0;
  int res = libsmctrl_get_tpc_info_cuda(&num_tpcs, 0);
  if (res != 0) {
    fprintf(stderr, "libsmctrl_get_tpc_info_cuda failed with %d\n", res);
    return 1;
  }
  int sms_per_tpc = (num_tpcs == 0) ? 0 : (num_sms / (int)num_tpcs);
  printf("Device: %d SMs, %u TPCs (~%d SM/TPC)\n", num_sms, num_tpcs, sms_per_tpc);
  printf("Method=%s windows=%d windows/phase=%d tpcs=%d<->%d blocks=%d repeats=%d\n",
         method.c_str(), windows, windows_per_phase, tpc_a, tpc_b, blocks, repeats);
  printf("ASCII timeline legend: '.' none, '+' light, '*' medium, '#' heavy block occupancy per SM\n");
  printf("Each row is one time window; columns map to SMID 0..%d\n\n", num_sms - 1);

  // A dedicated stream keeps sequencing explicit for mask->launch->sync.
  cudaStream_t stream;
  cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");

  // Device buffers:
  // - d_out: minimal per-block output sink
  // - d_smids: per-block SMID samples written by kernel
  float* d_out = nullptr;
  uint16_t* d_smids = nullptr;
  cuda_check(cudaMalloc(&d_out, sizeof(float) * blocks), "cudaMalloc d_out");
  cuda_check(cudaMalloc(&d_smids, sizeof(uint16_t) * blocks), "cudaMalloc d_smids");

  std::vector<WindowSummary> rows;
  rows.reserve(windows);

  // Core experiment loop:
  // - phase advances every windows_per_phase windows
  // - enabled TPC alternates between tpc_a and tpc_b by phase parity
  for (int w = 0; w < windows; w++) {
    int phase = w / windows_per_phase;
    int enabled_tpc = (phase % 2 == 0) ? tpc_a : tpc_b;
    WindowSummary s = run_window(w, phase, enabled_tpc, method, stream, blocks, repeats,
                                 num_sms, d_out, d_smids);
    print_ascii_timeline_row(s);
    rows.push_back(std::move(s));
  }

  write_csv(rows, csv_path);
  printf("\nWrote timeline CSV: %s\n", csv_path);
  printf("Plot with: python3 plot_dynamic_switch_timeline.py %s\n", csv_path);

  // Explicit teardown for clean exits and easier debugging.
  cuda_check(cudaFree(d_smids), "cudaFree d_smids");
  cuda_check(cudaFree(d_out), "cudaFree d_out");
  cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
  return 0;
}

