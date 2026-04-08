#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits.h>
#include <string>
#include <vector>
#include <unistd.h>

#include <cuda_runtime.h>

#include "libsmctrl.h"

// Runtime configuration for this test harness.
// The goal is to run two concurrent GEMM-like workloads (A and B), each pinned
// to a user-provided TPC set via libsmctrl stream masks.
// - tpcs_a / tpcs_b: comma/range syntax ("0,1,4-7")
// - m,n,k: GEMM dimensions
// - iters/warmup/repeat: controls kernel runtime and measurement stability
// - leak_thresh_pct: outside-partition threshold for PASS/WARN
// - csv_path: where to export structured metrics for plotting
struct Config {
  std::string tpcs_a;
  std::string tpcs_b;
  int m = 2048;
  int n = 2048;
  int k = 2048;
  int iters = 20;
  int warmup = 2;
  int repeat = 2;
  double leak_thresh_pct = 1.0;
  std::string csv_path;
};

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

static void usage(const char* argv0) {
  fprintf(stderr,
          "Usage: %s --tpcs-a <list> --tpcs-b <list> [--m M] [--n N] [--k K]\n"
          "          [--iters N] [--warmup N] [--repeat N] [--leak-thresh-pct P] [--csv PATH]\n"
          "\n"
          "TPC list format: comma-separated indexes and ranges, e.g. 0,1,4-7\n"
          "Example: %s --tpcs-a 0-29 --tpcs-b 30-59 --m 4096 --n 4096 --k 4096 --iters 30\n",
          argv0, argv0);
}

// Strict integer parser for CLI options.
// Returns false on malformed strings, overflow, or trailing characters.
static bool parse_int(const char* s, int* out) {
  errno = 0;
  char* end = nullptr;
  long v = std::strtol(s, &end, 10);
  if (errno != 0 || !end || *end != '\0') return false;
  if (v < INT32_MIN || v > INT32_MAX) return false;
  *out = (int)v;
  return true;
}

// Parse command-line arguments into a validated Config.
// We intentionally keep parsing explicit (instead of getopt) to keep behavior
// obvious for research scripts and reproducibility logs.
static Config parse_cli(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto need_val = [&](const char* flag) -> const char* {
      if (i + 1 >= argc) {
        fprintf(stderr, "missing value for %s\n", flag);
        usage(argv[0]);
        std::exit(1);
      }
      i++;
      return argv[i];
    };
    if (a == "--help" || a == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else if (a == "--tpcs-a") {
      cfg.tpcs_a = need_val("--tpcs-a");
    } else if (a == "--tpcs-b") {
      cfg.tpcs_b = need_val("--tpcs-b");
    } else if (a == "--m") {
      if (!parse_int(need_val("--m"), &cfg.m)) die("invalid --m");
    } else if (a == "--n") {
      if (!parse_int(need_val("--n"), &cfg.n)) die("invalid --n");
    } else if (a == "--k") {
      if (!parse_int(need_val("--k"), &cfg.k)) die("invalid --k");  
    } else if (a == "--iters") {
      if (!parse_int(need_val("--iters"), &cfg.iters)) die("invalid --iters");
    } else if (a == "--warmup") {
      if (!parse_int(need_val("--warmup"), &cfg.warmup)) die("invalid --warmup");
    } else if (a == "--repeat") {
      if (!parse_int(need_val("--repeat"), &cfg.repeat)) die("invalid --repeat");
    } else if (a == "--leak-thresh-pct") {
      cfg.leak_thresh_pct = std::atof(need_val("--leak-thresh-pct"));
    } else if (a == "--csv") {
      cfg.csv_path = need_val("--csv");
    } else {
      fprintf(stderr, "unknown argument: %s\n", a.c_str());
      usage(argv[0]);
      std::exit(1);
    }
  }
  if (cfg.tpcs_a.empty() || cfg.tpcs_b.empty()) {
    usage(argv[0]);
    die("both --tpcs-a and --tpcs-b are required");
  }
  if (cfg.m <= 0 || cfg.n <= 0 || cfg.k <= 0 || cfg.iters <= 0 || cfg.warmup < 0 || cfg.repeat <= 0) {
    die("matrix dimensions/iters/repeat must be positive and warmup must be >= 0");
  }
  return cfg;
}

// Utility helpers for output path resolution.
// We resolve relative CSV names against the executable directory so outputs
// always land in tests/parallel_partitioning even when launched from libsmctrl/.
static std::string dirname_of_path(const std::string& path) {
  size_t slash = path.find_last_of('/');
  if (slash == std::string::npos) return ".";
  if (slash == 0) return "/";
  return path.substr(0, slash);
}

static bool is_absolute_path(const std::string& path) {
  return !path.empty() && path[0] == '/';
}

static std::string resolve_output_csv_path(const std::string& requested_csv) {
  char exe_buf[PATH_MAX];
  ssize_t n = readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
  if (n <= 0) die("failed to resolve /proc/self/exe for output path");
  exe_buf[n] = '\0';
  std::string out_dir = dirname_of_path(std::string(exe_buf));
  if (requested_csv.empty()) {
    return out_dir + "/dual_tpc_gemm_metrics.csv";
  }
  if (is_absolute_path(requested_csv)) {
    return requested_csv;
  }
  // Relative output names are anchored to the binary directory (tests/parallel_partitioning).
  return out_dir + "/" + requested_csv;
}

// Parse a TPC set specification like "0,1,4-7" into sorted unique IDs.
// Validation is done against the discovered hardware TPC count.
// Duplicate values naturally collapse because we build through a bitmap first.
static std::vector<int> parse_tpc_set(const std::string& spec, uint32_t num_tpcs) {
  std::vector<int> selected(num_tpcs, 0);
  size_t pos = 0;
  while (pos < spec.size()) {
    size_t next = spec.find(',', pos);
    std::string tok = spec.substr(pos, (next == std::string::npos) ? std::string::npos : (next - pos));
    if (tok.empty()) die("empty token in TPC list");
    size_t dash = tok.find('-');
    if (dash == std::string::npos) {
      int idx = -1;
      if (!parse_int(tok.c_str(), &idx)) die("invalid TPC index in list");
      if (idx < 0 || idx >= (int)num_tpcs) die("TPC index out of range");
      selected[idx] = 1;
    } else {
      std::string a = tok.substr(0, dash);
      std::string b = tok.substr(dash + 1);
      int lo = -1, hi = -1;
      if (!parse_int(a.c_str(), &lo) || !parse_int(b.c_str(), &hi)) die("invalid TPC range in list");
      if (lo > hi) std::swap(lo, hi);
      if (lo < 0 || hi >= (int)num_tpcs) die("TPC range out of bounds");
      for (int t = lo; t <= hi; t++) selected[t] = 1;
    }
    if (next == std::string::npos) break;
    pos = next + 1;
  }
  std::vector<int> out;
  for (int t = 0; t < (int)num_tpcs; t++) {
    if (selected[t]) out.push_back(t);
  }
  if (out.empty()) die("TPC set must not be empty");
  return out;
}

// libsmctrl mask semantics: bit=1 means "disable this TPC".
// We start from an allow-list and invert it to produce the disable mask.
static uint128_t build_disable_mask_from_allowed(const std::vector<int>& tpcs) {
  uint128_t allow = 0;
  for (int t : tpcs) {
    allow |= ((uint128_t)1 << t);
  }
  return ~allow;
}

// Human-readable set printing for logs.
static std::string join_tpcs(const std::vector<int>& tpcs) {
  std::string s;
  for (size_t i = 0; i < tpcs.size(); i++) {
    if (i) s += ",";
    s += std::to_string(tpcs[i]);
  }
  return s;
}

// Device-side initializer to avoid host->device memcpy for large test matrices.
// Slightly perturbed values avoid degenerate constant behavior.
__global__ void init_vec(float* data, int n, float val) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] = val + (float)(i & 7) * 1e-4f;
}

// GEMM-like probe kernel.
//
// Two responsibilities are combined:
// 1) Do substantial compute work (matrix multiply inner loops) so concurrent
//    stream scheduling behavior is observable.
// 2) Instrument placement by reading %smid once per block and incrementing
//    an SM histogram.
//
// Important nuance:
// - "blocks" in reported metrics means CUDA thread blocks (CTAs), not SMs/TPCs.
// - We count blocks seen on each SM; later we infer TPC histograms from SM IDs.
__global__ void gemm_probe_kernel(const float* A, const float* B, float* C,
                                  int m, int n, int k, int repeat,
                                  int* sm_hist, uint16_t* smid_per_block) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int smid = 0;
  asm("mov.u32 %0, %%smid;" : "=r"(smid));
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    atomicAdd(&sm_hist[smid], 1);
    smid_per_block[blockIdx.y * gridDim.x + blockIdx.x] = (uint16_t)smid;
  }

  if (row >= m || col >= n) return;
  float acc = 0.0f;
  for (int r = 0; r < repeat; r++) {
    float local = 0.0f;
    for (int kk = 0; kk < k; kk++) {
      local += A[row * k + kk] * B[kk * n + col];
    }
    acc += local * 1.000001f + 0.00001f;
  }
  C[row * n + col] = acc;
}

// Per-workload runtime state.
// Each workload owns a stream, matrix buffers, and instrumentation buffers.
// A and B share the same process/context but have different stream masks.
struct Workload {
  const char* label;
  std::vector<int> allowed_tpcs;
  uint128_t disable_mask;
  cudaStream_t stream = nullptr;
  float *A = nullptr, *B = nullptr, *C = nullptr;
  int* d_sm_hist = nullptr;
  uint16_t* d_smid_per_block = nullptr;
  std::vector<int> h_sm_hist;
  std::vector<uint16_t> h_smid_per_block;
  float elapsed_ms = 0.0f;
};

// Map SM -> TPC using the same simple heuristic as the dynamic-switch demo.
// This is architecture-dependent; warnings are emitted when mapping quality is
// uncertain (e.g., non-even SM/TPC ratio).
static int tpc_of_sm(int smid, int sms_per_tpc, uint32_t num_tpcs) {
  if (sms_per_tpc <= 0 || smid < 0) return -1;
  int tpc = smid / sms_per_tpc;
  if (tpc < 0 || tpc >= (int)num_tpcs) return -1;
  return tpc;
}

// Compact per-SM activity row for quick terminal sanity checks.
static void print_sm_hist_ascii(const std::vector<int>& hist) {
  printf("  SM activity: |");
  for (size_t i = 0; i < hist.size(); i++) {
    char c = '.';
    if (hist[i] >= 32) c = '#';
    else if (hist[i] >= 8) c = '*';
    else if (hist[i] > 0) c = '+';
    putchar(c);
  }
  printf("|\n");
}

static double pct(uint64_t part, uint64_t total) {
  if (total == 0) return 0.0;
  return (100.0 * (double)part) / (double)total;
}

// Summarize one workload's observed execution placement.
//
// Outputs:
// - inside/outside percentages against this workload's requested TPC set
// - cross percentage (how much landed in the other workload's set)
// - inferred per-TPC histogram and compact SM activity line
//
// This is the core verification path proving whether partitioning is respected.
static void summarize_workload(const Workload& w,
                               const std::vector<int>& other_set_bitmap,
                               uint32_t num_tpcs,
                               int num_sms,
                               int sms_per_tpc,
                               double* outside_pct,
                               double* cross_pct,
                               uint64_t* bad_smid_count) {
  std::vector<uint64_t> tpc_hist(num_tpcs, 0);
  std::vector<int> own_set_bitmap(num_tpcs, 0);
  for (int t : w.allowed_tpcs) own_set_bitmap[t] = 1;

  uint64_t total_blocks = 0;
  uint64_t inside = 0;
  uint64_t outside = 0;
  uint64_t cross = 0;
  uint64_t invalid = 0;
  for (int sm = 0; sm < num_sms; sm++) {
    int blocks = w.h_sm_hist[sm];
    if (blocks <= 0) continue;
    total_blocks += (uint64_t)blocks;
    int tpc = tpc_of_sm(sm, sms_per_tpc, num_tpcs);
    if (tpc < 0) {
      invalid += (uint64_t)blocks;
      continue;
    }
    tpc_hist[tpc] += (uint64_t)blocks;
    if (own_set_bitmap[tpc]) inside += (uint64_t)blocks;
    else outside += (uint64_t)blocks;
    if (other_set_bitmap[tpc]) cross += (uint64_t)blocks;
  }

  *outside_pct = pct(outside, total_blocks);
  *cross_pct = pct(cross, total_blocks);
  *bad_smid_count = invalid;

  printf("\n[%s] requested_tpcs={%s}\n", w.label, join_tpcs(w.allowed_tpcs).c_str());
  printf("  elapsed_ms=%.3f total_observed_blocks=%llu inside=%llu (%.2f%%) outside=%llu (%.2f%%)\n",
         w.elapsed_ms,
         (unsigned long long)total_blocks,
         (unsigned long long)inside, pct(inside, total_blocks),
         (unsigned long long)outside, *outside_pct);
  printf("  on_other_workload_tpcs=%llu (%.2f%%)\n",
         (unsigned long long)cross, *cross_pct);
  if (invalid > 0) {
    printf("  warning: observed %llu block samples with invalid SM->TPC mapping\n",
           (unsigned long long)invalid);
  }
  print_sm_hist_ascii(w.h_sm_hist);

  printf("  inferred TPC histogram:");
  for (uint32_t t = 0; t < num_tpcs; t++) {
    if (tpc_hist[t] > 0) printf(" tpc%d=%llu", t, (unsigned long long)tpc_hist[t]);
  }
  printf("\n");
}

// Export machine-readable metrics for plotting.
//
// CSV schema:
// - kind=sm_hist: one row per (workload, smid)
// - kind=tpc_hist: one row per (workload, tpc)
//
// Keeping both granularities lets the plotter show raw SM occupancy and
// partition-level behavior without re-running the benchmark.
static void write_metrics_csv(const char* path,
                              const Workload& A,
                              const Workload& B,
                              uint32_t num_tpcs,
                              int num_sms,
                              int sms_per_tpc,
                              int pid) {
  FILE* f = fopen(path, "w");
  if (!f) die("unable to open CSV output path");

  fprintf(f, "kind,workload,pid,smid,tpc,blocks,elapsed_ms\n");
  const Workload* ws[2] = {&A, &B};
  for (const Workload* w : ws) {
    for (int sm = 0; sm < num_sms; sm++) {
      int blocks = w->h_sm_hist[sm];
      int tpc = tpc_of_sm(sm, sms_per_tpc, num_tpcs);
      fprintf(f, "sm_hist,%s,%d,%d,%d,%d,%.6f\n",
              w->label, pid, sm, tpc, blocks, w->elapsed_ms);
    }
    std::vector<uint64_t> tpc_hist(num_tpcs, 0);
    for (int sm = 0; sm < num_sms; sm++) {
      int tpc = tpc_of_sm(sm, sms_per_tpc, num_tpcs);
      if (tpc < 0 || tpc >= (int)num_tpcs) continue;
      int blocks = w->h_sm_hist[sm];
      if (blocks > 0) tpc_hist[tpc] += (uint64_t)blocks;
    }
    for (uint32_t t = 0; t < num_tpcs; t++) {
      fprintf(f, "tpc_hist,%s,%d,-1,%u,%llu,%.6f\n",
              w->label, pid, t, (unsigned long long)tpc_hist[t], w->elapsed_ms);
    }
  }
  fclose(f);
}

int main(int argc, char** argv) {
  // 1) Parse config and anchor output path.
  Config cfg = parse_cli(argc, argv);
  cfg.csv_path = resolve_output_csv_path(cfg.csv_path);

  // 2) Discover topology needed for validation and SM->TPC inference.
  int num_sms = 0;
  cuda_check(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0),
             "cudaDeviceGetAttribute(MP count)");
  uint32_t num_tpcs = 0;
  int lres = libsmctrl_get_tpc_info_cuda(&num_tpcs, 0);
  if (lres != 0) {
    fprintf(stderr, "libsmctrl_get_tpc_info_cuda failed with %d\n", lres);
    return 1;
  }
  int sms_per_tpc = (num_tpcs == 0) ? 0 : (num_sms / (int)num_tpcs);
  if (sms_per_tpc <= 0) die("unable to derive sms_per_tpc");

  // 3) Parse requested partitions and detect overlap.
  std::vector<int> set_a = parse_tpc_set(cfg.tpcs_a, num_tpcs);
  std::vector<int> set_b = parse_tpc_set(cfg.tpcs_b, num_tpcs);
  std::vector<int> bitmap_a(num_tpcs, 0), bitmap_b(num_tpcs, 0);
  for (int t : set_a) bitmap_a[t] = 1;
  for (int t : set_b) bitmap_b[t] = 1;

  int overlap = 0;
  for (uint32_t t = 0; t < num_tpcs; t++) overlap += (bitmap_a[t] && bitmap_b[t]) ? 1 : 0;

  printf("PID=%d device_sms=%d device_tpcs=%u sms_per_tpc=%d\n",
         (int)getpid(), num_sms, num_tpcs, sms_per_tpc);
  printf("WorkloadA tpcs={%s}\n", join_tpcs(set_a).c_str());
  printf("WorkloadB tpcs={%s}\n", join_tpcs(set_b).c_str());
  if (num_sms % (int)num_tpcs != 0) {
    printf("warning: num_sms is not evenly divisible by num_tpcs, inferred SM->TPC mapping is approximate\n");
  }
  if (overlap > 0) {
    printf("warning: requested sets overlap on %d TPC(s); isolation metrics will reflect that overlap\n", overlap);
  }

  // 4) Prepare launch geometry and two workload descriptors.
  const int size_A = cfg.m * cfg.k;
  const int size_B = cfg.k * cfg.n;
  const int size_C = cfg.m * cfg.n;
  const dim3 block(16, 16);
  const dim3 grid((cfg.n + block.x - 1) / block.x, (cfg.m + block.y - 1) / block.y);
  const int blocks_total = (int)(grid.x * grid.y);

  Workload A;
  A.label = "A";
  A.allowed_tpcs = set_a;
  A.disable_mask = build_disable_mask_from_allowed(set_a);
  Workload B;
  B.label = "B";
  B.allowed_tpcs = set_b;
  B.disable_mask = build_disable_mask_from_allowed(set_b);

  // 5) Per-workload setup:
  //    - create stream
  //    - apply stream-scoped libsmctrl mask
  //    - allocate matrices + instrumentation buffers
  //    - initialize A/B matrices on-device
  auto setup_workload = [&](Workload* w, float seed) {
    cuda_check(cudaStreamCreate(&w->stream), "cudaStreamCreate");
    libsmctrl_set_stream_mask_ext((void*)w->stream, w->disable_mask);
    cuda_check(cudaMalloc(&w->A, sizeof(float) * size_A), "cudaMalloc A");
    cuda_check(cudaMalloc(&w->B, sizeof(float) * size_B), "cudaMalloc B");
    cuda_check(cudaMalloc(&w->C, sizeof(float) * size_C), "cudaMalloc C");
    cuda_check(cudaMalloc(&w->d_sm_hist, sizeof(int) * num_sms), "cudaMalloc d_sm_hist");
    cuda_check(cudaMalloc(&w->d_smid_per_block, sizeof(uint16_t) * blocks_total), "cudaMalloc d_smid_per_block");
    cuda_check(cudaMemsetAsync(w->d_sm_hist, 0, sizeof(int) * num_sms, w->stream), "cudaMemsetAsync d_sm_hist");
    const int tpb = 256;
    const int blocksA = (size_A + tpb - 1) / tpb;
    const int blocksB = (size_B + tpb - 1) / tpb;
    init_vec<<<blocksA, tpb, 0, w->stream>>>(w->A, size_A, seed);
    init_vec<<<blocksB, tpb, 0, w->stream>>>(w->B, size_B, seed + 1.0f);
    cuda_check(cudaGetLastError(), "init_vec launch");
  };

  setup_workload(&A, 1.0f);
  setup_workload(&B, 2.0f);
  cuda_check(cudaStreamSynchronize(A.stream), "sync A after init");
  cuda_check(cudaStreamSynchronize(B.stream), "sync B after init");

  // 6) Warmup to reduce first-launch effects, then reset histograms.
  for (int i = 0; i < cfg.warmup; i++) {
    gemm_probe_kernel<<<grid, block, 0, A.stream>>>(A.A, A.B, A.C, cfg.m, cfg.n, cfg.k, cfg.repeat,
                                                     A.d_sm_hist, A.d_smid_per_block);
    gemm_probe_kernel<<<grid, block, 0, B.stream>>>(B.A, B.B, B.C, cfg.m, cfg.n, cfg.k, cfg.repeat,
                                                     B.d_sm_hist, B.d_smid_per_block);
  }
  cuda_check(cudaGetLastError(), "warmup launch");
  cuda_check(cudaStreamSynchronize(A.stream), "sync A warmup");
  cuda_check(cudaStreamSynchronize(B.stream), "sync B warmup");
  cuda_check(cudaMemsetAsync(A.d_sm_hist, 0, sizeof(int) * num_sms, A.stream), "reset A hist");
  cuda_check(cudaMemsetAsync(B.d_sm_hist, 0, sizeof(int) * num_sms, B.stream), "reset B hist");
  cuda_check(cudaStreamSynchronize(A.stream), "sync A reset");
  cuda_check(cudaStreamSynchronize(B.stream), "sync B reset");

  // 7) Record independent stream durations and enqueue concurrent work.
  // We interleave A and B launches to maximize overlap opportunities.
  cudaEvent_t a_start, a_end, b_start, b_end;
  cuda_check(cudaEventCreate(&a_start), "cudaEventCreate a_start");
  cuda_check(cudaEventCreate(&a_end), "cudaEventCreate a_end");
  cuda_check(cudaEventCreate(&b_start), "cudaEventCreate b_start");
  cuda_check(cudaEventCreate(&b_end), "cudaEventCreate b_end");

  cuda_check(cudaEventRecord(a_start, A.stream), "record a_start");
  cuda_check(cudaEventRecord(b_start, B.stream), "record b_start");
  for (int i = 0; i < cfg.iters; i++) {
    gemm_probe_kernel<<<grid, block, 0, A.stream>>>(A.A, A.B, A.C, cfg.m, cfg.n, cfg.k, cfg.repeat,
                                                     A.d_sm_hist, A.d_smid_per_block);
    gemm_probe_kernel<<<grid, block, 0, B.stream>>>(B.A, B.B, B.C, cfg.m, cfg.n, cfg.k, cfg.repeat,
                                                     B.d_sm_hist, B.d_smid_per_block);
  } 
  cuda_check(cudaGetLastError(), "measurement launch");
  cuda_check(cudaEventRecord(a_end, A.stream), "record a_end");
  cuda_check(cudaEventRecord(b_end, B.stream), "record b_end");

  // 8) Join streams, collect timings, and copy instrumentation buffers.
  cuda_check(cudaStreamSynchronize(A.stream), "sync A measurement");
  cuda_check(cudaStreamSynchronize(B.stream), "sync B measurement");
  cuda_check(cudaEventElapsedTime(&A.elapsed_ms, a_start, a_end), "elapsed A");
  cuda_check(cudaEventElapsedTime(&B.elapsed_ms, b_start, b_end), "elapsed B");

  A.h_sm_hist.assign(num_sms, 0);
  B.h_sm_hist.assign(num_sms, 0);
  A.h_smid_per_block.assign(blocks_total, 0);
  B.h_smid_per_block.assign(blocks_total, 0);
  cuda_check(cudaMemcpy(A.h_sm_hist.data(), A.d_sm_hist, sizeof(int) * num_sms, cudaMemcpyDeviceToHost),
             "copy A hist");
  cuda_check(cudaMemcpy(B.h_sm_hist.data(), B.d_sm_hist, sizeof(int) * num_sms, cudaMemcpyDeviceToHost),
             "copy B hist");
  cuda_check(cudaMemcpy(A.h_smid_per_block.data(), A.d_smid_per_block, sizeof(uint16_t) * blocks_total, cudaMemcpyDeviceToHost),
             "copy A smids");
  cuda_check(cudaMemcpy(B.h_smid_per_block.data(), B.d_smid_per_block, sizeof(uint16_t) * blocks_total, cudaMemcpyDeviceToHost),
             "copy B smids");

  // 9) Print concise run metadata + per-workload verification summaries.
  printf("\nRun config: m=%d n=%d k=%d iters=%d warmup=%d repeat=%d grid=(%u,%u) block=(%u,%u)\n",
         cfg.m, cfg.n, cfg.k, cfg.iters, cfg.warmup, cfg.repeat, grid.x, grid.y, block.x, block.y);
  printf("Process attribution: pid=%d streamA=%p streamB=%p kernel=gemm_probe_kernel\n",
         (int)getpid(), (void*)A.stream, (void*)B.stream);

  double outside_a = 0.0, outside_b = 0.0;
  double cross_a = 0.0, cross_b = 0.0;
  uint64_t bad_a = 0, bad_b = 0;
  summarize_workload(A, bitmap_b, num_tpcs, num_sms, sms_per_tpc, &outside_a, &cross_a, &bad_a);
  summarize_workload(B, bitmap_a, num_tpcs, num_sms, sms_per_tpc, &outside_b, &cross_b, &bad_b);

  // 10) Emit CSV for matplotlib pipeline and produce final verdict.
  const bool pass = (outside_a <= cfg.leak_thresh_pct) && (outside_b <= cfg.leak_thresh_pct) && (bad_a == 0) && (bad_b == 0);
  write_metrics_csv(cfg.csv_path.c_str(), A, B, num_tpcs, num_sms, sms_per_tpc, (int)getpid());
  printf("Wrote metrics CSV: %s\n", cfg.csv_path.c_str());
  printf("Plot with: python3 plot_dual_tpc_gemm.py %s\n", cfg.csv_path.c_str());
  printf("\nVerdict: %s (outside-set threshold %.2f%%, A=%.2f%%, B=%.2f%%)\n",
         pass ? "PASS" : "WARN",
         cfg.leak_thresh_pct, outside_a, outside_b);

  // 11) Explicit teardown for predictable behavior in repeated runs.
  auto teardown = [&](Workload* w) {
    if (w->d_smid_per_block) cudaFree(w->d_smid_per_block);
    if (w->d_sm_hist) cudaFree(w->d_sm_hist);
    if (w->C) cudaFree(w->C);
    if (w->B) cudaFree(w->B);
    if (w->A) cudaFree(w->A);
    if (w->stream) cudaStreamDestroy(w->stream);
  };
  teardown(&A);
  teardown(&B);
  cudaEventDestroy(a_start);
  cudaEventDestroy(a_end);
  cudaEventDestroy(b_start);
  cudaEventDestroy(b_end);
  return pass ? 0 : 2;
}
