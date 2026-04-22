#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <limits.h>
#include <string>
#include <sstream>
#include <deque>
#include <unistd.h>
#include <vector>

#include <cuda_runtime.h>

#include "libsmctrl.h"

// Runtime knobs: schedule shape, sampling cadence, and workload size.
struct Config {
  std::string sm_ranges_spec;
  // Maximum number of stream+mask slices to run (safety cap).
  // If the GEMM finishes earlier, the run ends naturally.
  int max_slices = 1000000;
  int warmup = 5;
  int hold_iters = 1;
  int sample_every = 1;
  // Number of output rows computed per slice (one fresh stream per slice).
  int slice_rows = 256;
  // Maximum number of concurrently in-flight streams/tiles. When exceeded, the
  // oldest completed stream is drained and destroyed to keep enqueue pressure high.
  int max_inflight = 4096;
  int m = 2048;
  int n = 2048;
  int k = 2048;
  std::string csv_prefix = "single_process_dynamic_mask_matmul";
};

// Per-iteration record used by plots and aggregate summaries.
struct IterRow {
  int iter = 0;
  int mask_id = -1;
  int expected_sm_lo = -1;
  int expected_sm_hi = -1;
  int dominant_tpc = -1;
  double switch_us = 0.0;
  double launch_us = 0.0;
  double set_plus_launch_us = 0.0;
  int sampled = 0;
  int blocks_total = 0;
  int inside_blocks = 0;
  int outside_blocks = 0;
  int invalid_blocks = 0;
  double inside_pct = 0.0;
  double outside_pct = 0.0;
  int smid_min = -1;
  int smid_max = -1;
  int distinct_smid_count = 0;
  std::string observed_smids;
  int mismatch = 0;
};

struct SmRange {
  int lo = 0;
  int hi = 0;
  uint128_t disable_mask = ~((uint128_t)0);
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

// Monotonic host timer for all latency/cadence measurements.
static uint64_t now_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static bool parse_int(const char* s, int* out) {
  errno = 0;
  char* end = nullptr;
  long v = strtol(s, &end, 10);
  if (errno != 0 || !end || *end != '\0') return false;
  if (v < INT32_MIN || v > INT32_MAX) return false;
  *out = (int)v;
  return true;
}

static void usage(const char* argv0) {
  fprintf(stderr,
          "Usage: %s [--sm-ranges list] [--max-slices N] [--warmup N] [--hold-iters N]\n"
          "          [--sample-every N] [--slice-rows N] [--max-inflight N]\n"
          "          [--m N] [--n N] [--k N] [--csv-prefix name]\n"
          "  --sm-ranges: comma-separated SM ranges, e.g. 5-10,11-25,26-40\n",
          argv0);
}

static Config parse_cli(int argc, char** argv) {
  Config c;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto need = [&](const char* flag) -> const char* {
      if (i + 1 >= argc) {
        fprintf(stderr, "missing value for %s\n", flag);
        std::exit(1);
      }
      i++;
      return argv[i];
    };
    if (a == "--help" || a == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else if (a == "--sm-ranges") {
      c.sm_ranges_spec = need("--sm-ranges");
    } else if (a == "--max-slices") {
      if (!parse_int(need("--max-slices"), &c.max_slices)) die("invalid --max-slices");
    } else if (a == "--warmup") {
      if (!parse_int(need("--warmup"), &c.warmup)) die("invalid --warmup");
    } else if (a == "--hold-iters") {
      if (!parse_int(need("--hold-iters"), &c.hold_iters)) die("invalid --hold-iters");
    } else if (a == "--sample-every") {
      if (!parse_int(need("--sample-every"), &c.sample_every)) die("invalid --sample-every");
    } else if (a == "--slice-rows") {
      if (!parse_int(need("--slice-rows"), &c.slice_rows)) die("invalid --slice-rows");
    } else if (a == "--max-inflight") {
      if (!parse_int(need("--max-inflight"), &c.max_inflight)) die("invalid --max-inflight");
    } else if (a == "--m") {
      if (!parse_int(need("--m"), &c.m)) die("invalid --m");
    } else if (a == "--n") {
      if (!parse_int(need("--n"), &c.n)) die("invalid --n");
    } else if (a == "--k") {
      if (!parse_int(need("--k"), &c.k)) die("invalid --k");
    } else if (a == "--csv-prefix") {
      c.csv_prefix = need("--csv-prefix");
    } else {
      usage(argv[0]);
      die("unknown argument");
    }
  }

  if (c.max_slices <= 0 || c.warmup < 0 || c.hold_iters <= 0 || c.sample_every <= 0 || c.slice_rows <= 0 ||
      c.max_inflight <= 0) {
    die("invalid numeric options");
  }
  if (c.m <= 0 || c.n <= 0 || c.k <= 0) {
    die("matrix dimensions must be positive");
  }
  return c;
}

static std::string dirname_of_path(const std::string& path) {
  size_t slash = path.find_last_of('/');
  if (slash == std::string::npos) return ".";
  if (slash == 0) return "/";
  return path.substr(0, slash);
}

static std::string resolve_out_dir_from_exe() {
  char exe_buf[PATH_MAX];
  ssize_t n = readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
  if (n <= 0) die("failed to resolve /proc/self/exe");
  exe_buf[n] = '\0';
  return dirname_of_path(std::string(exe_buf));
}

// Parse comma-separated SM ranges ("5-10,11-25") and build libsmctrl TPC masks.
static std::vector<SmRange> parse_sm_ranges(const std::string& spec,
                                            int num_sms,
                                            int sms_per_tpc,
                                            uint32_t num_tpcs) {
  if (num_sms <= 0 || sms_per_tpc <= 0 || num_tpcs == 0) die("invalid topology for SM-range parsing");
  if (spec.empty()) die("--sm-ranges must be provided");

  auto build_disable_mask_for_sm_range = [&](int lo, int hi) {
    uint128_t enable_mask = 0;
    for (int sm = lo; sm <= hi; sm++) {
      int tpc = sm / sms_per_tpc;
      if (tpc >= 0 && tpc < (int)num_tpcs) {
        enable_mask |= ((uint128_t)1 << tpc);
      }
    }
    // libsmctrl semantics: 1 bit means disabled.
    return ~enable_mask;
  };

  std::vector<SmRange> out;
  size_t pos = 0;
  while (pos < spec.size()) {
    size_t next = spec.find(',', pos);
    std::string tok = spec.substr(pos, (next == std::string::npos) ? std::string::npos : (next - pos));
    if (tok.empty()) die("empty token in --sm-ranges");
    size_t dash = tok.find('-');
    if (dash == std::string::npos) die("each --sm-ranges token must be lo-hi");
    int lo = -1, hi = -1;
    std::string a = tok.substr(0, dash);
    std::string b = tok.substr(dash + 1);
    if (!parse_int(a.c_str(), &lo) || !parse_int(b.c_str(), &hi)) die("invalid --sm-ranges token");
    if (lo > hi) std::swap(lo, hi);
    if (lo < 0 || hi >= num_sms) die("SM range out of bounds");
    SmRange r;
    r.lo = lo;
    r.hi = hi;
    r.disable_mask = build_disable_mask_for_sm_range(lo, hi);
    out.push_back(r);
    if (next == std::string::npos) break;
    pos = next + 1;
  }
  if (out.empty()) die("no SM ranges selected");
  return out;
}

static std::string format_distinct_smids(const std::vector<int>& sm_seen) {
  std::ostringstream oss;
  bool first = true;
  for (size_t sm = 0; sm < sm_seen.size(); sm++) {
    if (!sm_seen[sm]) continue;
    if (!first) oss << ";";
    oss << sm;
    first = false;
  }
  return oss.str();
}

static int percentile_index(int n, double p) {
  if (n <= 0) return 0;
  int idx = (int)(p * (double)(n - 1));
  if (idx < 0) idx = 0;
  if (idx >= n) idx = n - 1;
  return idx;
}

static double pct(int a, int total) {
  if (total <= 0) return 0.0;
  return 100.0 * (double)a / (double)total;
}

static int tpc_of_sm(int sm, int sms_per_tpc, uint32_t num_tpcs) {
  if (sm < 0 || sms_per_tpc <= 0) return -1;
  int tpc = sm / sms_per_tpc;
  if (tpc < 0 || tpc >= (int)num_tpcs) return -1;
  return tpc;
}

// Compute one slice of C = A*B for rows [row0, row0 + m_slice).
// Also stamps one SMID per block for residency analysis.
__global__ void tiled_matmul_slice_with_smid(const float* a,
                                             const float* b,
                                             float* c,
                                             int* block_smid,
                                             int row0,
                                             int m_slice,
                                             int m_total,
                                             int n,
                                             int k) {
  __shared__ float as[16][16];
  __shared__ float bs[16][16];

  int row = row0 + blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float acc = 0.0f;

  int smid = 0;
  asm("mov.u32 %0, %%smid;" : "=r"(smid));
  if (block_smid != nullptr && threadIdx.x == 0 && threadIdx.y == 0) {
    block_smid[blockIdx.y * gridDim.x + blockIdx.x] = smid;
  }

  for (int tile = 0; tile < (k + 15) / 16; tile++) {
    int a_col = tile * 16 + threadIdx.x;
    int b_row = tile * 16 + threadIdx.y;

    // A is logically m_total x k; we only compute a slice of rows.
    as[threadIdx.y][threadIdx.x] = (row < m_total && a_col < k) ? a[row * k + a_col] : 0.0f;
    bs[threadIdx.y][threadIdx.x] = (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;
    __syncthreads();

    for (int kk = 0; kk < 16; kk++) {
      acc += as[threadIdx.y][kk] * bs[kk][threadIdx.x];
    }
    __syncthreads();
  }

  if ((row - row0) < m_slice && row < m_total && col < n) {
    c[row * n + col] = acc;
  }
}

static void write_iterations_csv(const std::string& path, const std::vector<IterRow>& rows) {
  FILE* f = fopen(path.c_str(), "w");
  if (!f) die("failed to open iterations csv");
  fprintf(f,
          "iter,mask_id,expected_sm_lo,expected_sm_hi,dominant_tpc,switch_us,launch_us,set_plus_launch_us,"
          "sampled,blocks_total,inside_blocks,outside_blocks,invalid_blocks,inside_pct,outside_pct,"
          "smid_min,smid_max,distinct_smid_count,observed_smids,mismatch\n");
  for (const auto& r : rows) {
    fprintf(f,
            "%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%d,%d,%d,%d,%d,%.6f,%.6f,%d,%d,%d,%s,%d\n",
            r.iter, r.mask_id, r.expected_sm_lo, r.expected_sm_hi, r.dominant_tpc, r.switch_us, r.launch_us,
            r.set_plus_launch_us, r.sampled, r.blocks_total, r.inside_blocks, r.outside_blocks,
            r.invalid_blocks, r.inside_pct, r.outside_pct, r.smid_min, r.smid_max,
            r.distinct_smid_count, r.observed_smids.c_str(), r.mismatch);
  }
  fclose(f);
}

static void write_summary_csv(const std::string& path,
                              int iters,
                              int warmup,
                              int hold_iters,
                              int sample_every,
                              int slice_rows,
                              int max_inflight,
                              int m,
                              int n,
                              int k,
                              int schedule_len,
                              int sampled_iters,
                              double switches_per_sec,
                              double mean_sw,
                              double med_sw,
                              double p95_sw,
                              double p99_sw,
                              double mean_total,
                              double med_total,
                              double p95_total,
                              double p99_total,
                              int mismatch_count,
                              double avg_outside_pct) {
  FILE* f = fopen(path.c_str(), "w");
  if (!f) die("failed to open summary csv");
  fprintf(f, "metric,value\n");
  fprintf(f, "iterations,%d\n", iters);
  fprintf(f, "warmup,%d\n", warmup);
  fprintf(f, "hold_iters,%d\n", hold_iters);
  fprintf(f, "sample_every,%d\n", sample_every);
  fprintf(f, "slice_rows,%d\n", slice_rows);
  fprintf(f, "max_inflight,%d\n", max_inflight);
  fprintf(f, "matrix_m,%d\n", m);
  fprintf(f, "matrix_n,%d\n", n);
  fprintf(f, "matrix_k,%d\n", k);
  fprintf(f, "schedule_len,%d\n", schedule_len);
  fprintf(f, "sampled_iterations,%d\n", sampled_iters);
  fprintf(f, "switches_per_sec,%.6f\n", switches_per_sec);
  fprintf(f, "switch_us_mean,%.6f\n", mean_sw);
  fprintf(f, "switch_us_median,%.6f\n", med_sw);
  fprintf(f, "switch_us_p95,%.6f\n", p95_sw);
  fprintf(f, "switch_us_p99,%.6f\n", p99_sw);
  fprintf(f, "set_plus_launch_us_mean,%.6f\n", mean_total);
  fprintf(f, "set_plus_launch_us_median,%.6f\n", med_total);
  fprintf(f, "set_plus_launch_us_p95,%.6f\n", p95_total);
  fprintf(f, "set_plus_launch_us_p99,%.6f\n", p99_total);
  fprintf(f, "mismatch_count,%d\n", mismatch_count);
  fprintf(f, "avg_outside_pct,%.6f\n", avg_outside_pct);
  fclose(f);
}

struct InflightSlice {
  cudaStream_t stream{};
  cudaEvent_t done{};
  int deploy_idx = -1;
  int schedule_idx = -1;
  int expected_lo = -1;
  int expected_hi = -1;
  int row0 = 0;
  int m_slice = 0;
  int blocks_total = 0;
  bool sampled = false;
  int* d_smid = nullptr;

  double switch_us = 0.0;
  double launch_us = 0.0;
  double set_plus_launch_us = 0.0;
};

static void finish_inflight_slice(InflightSlice& w,
                                  int num_sms,
                                  int sms_per_tpc,
                                  uint32_t num_tpcs,
                                  std::vector<IterRow>& rows,
                                  int* sampled_iters) {
  cuda_check(cudaEventSynchronize(w.done), "cudaEventSynchronize inflight");

  IterRow row;
  row.iter = w.deploy_idx;
  row.mask_id = w.schedule_idx;
  row.expected_sm_lo = w.expected_lo;
  row.expected_sm_hi = w.expected_hi;
  row.switch_us = w.switch_us;
  row.launch_us = w.launch_us;
  row.set_plus_launch_us = w.set_plus_launch_us;
  row.sampled = w.sampled ? 1 : 0;

  if (w.sampled) {
    (*sampled_iters)++;
    row.blocks_total = w.blocks_total;

    size_t smid_bytes = (size_t)w.blocks_total * sizeof(int);
    std::vector<int> h_block_smid;
    h_block_smid.resize(w.blocks_total);
    cuda_check(cudaMemcpy(h_block_smid.data(), w.d_smid, smid_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy sampled smid");

    std::vector<int> tpc_counts(num_tpcs, 0);
    std::vector<int> sm_seen(num_sms, 0);

    int smid_min = INT32_MAX;
    int smid_max = INT32_MIN;
    int inside = 0;
    int outside = 0;
    int invalid = 0;
    int distinct_smid_count = 0;

    for (int bidx = 0; bidx < w.blocks_total; bidx++) {
      int sm = h_block_smid[bidx];
      if (sm >= 0 && sm < num_sms) {
        if (sm_seen[sm] == 0) {
          sm_seen[sm] = 1;
          distinct_smid_count++;
        }
        if (sm < smid_min) smid_min = sm;
        if (sm > smid_max) smid_max = sm;
      }

      int obs_tpc = tpc_of_sm(sm, sms_per_tpc, num_tpcs);
      if (obs_tpc < 0) {
        invalid++;
      } else {
        tpc_counts[obs_tpc]++;
        if (sm >= w.expected_lo && sm <= w.expected_hi) {
          inside++;
        } else {
          outside++;
        }
      }
    }

    int dominant_tpc = -1;
    int dominant_cnt = -1;
    for (uint32_t t = 0; t < num_tpcs; t++) {
      if (tpc_counts[t] > dominant_cnt) {
        dominant_cnt = tpc_counts[t];
        dominant_tpc = (int)t;
      }
    }

    row.dominant_tpc = dominant_tpc;
    row.inside_blocks = inside;
    row.outside_blocks = outside;
    row.invalid_blocks = invalid;
    row.inside_pct = pct(inside, inside + outside);
    row.outside_pct = pct(outside, inside + outside);
    row.smid_min = (smid_min == INT32_MAX) ? -1 : smid_min;
    row.smid_max = (smid_max == INT32_MIN) ? -1 : smid_max;
    row.distinct_smid_count = distinct_smid_count;
    row.observed_smids = format_distinct_smids(sm_seen);
    row.mismatch = (outside > 0) ? 1 : 0;

    cuda_check(cudaFree(w.d_smid), "cudaFree sampled d_smid");
    w.d_smid = nullptr;
  }

  rows.push_back(row);

  cuda_check(cudaEventDestroy(w.done), "cudaEventDestroy inflight");
  cuda_check(cudaStreamDestroy(w.stream), "cudaStreamDestroy inflight");
}

int main(int argc, char** argv) {
  // 1) Parse CLI and discover device topology needed for SM->TPC inference.
  Config cfg = parse_cli(argc, argv);

  int num_sms = 0;
  cuda_check(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0),
             "cudaDeviceGetAttribute(MP count)");
  uint32_t num_tpcs = 0;
  int res = libsmctrl_get_tpc_info_cuda(&num_tpcs, 0);
  if (res != 0) die("libsmctrl_get_tpc_info_cuda failed");
  int sms_per_tpc = (num_tpcs == 0) ? 0 : (num_sms / (int)num_tpcs);
  if (sms_per_tpc <= 0) die("invalid sms_per_tpc");

  // 2) Build switching schedule and derived launch geometry.
  std::vector<SmRange> schedule = parse_sm_ranges(cfg.sm_ranges_spec, num_sms, sms_per_tpc, num_tpcs);
  int blocks_x = (cfg.n + 15) / 16;
  int blocks_y_max = (cfg.slice_rows + 15) / 16;
  int blocks_total_max = blocks_x * blocks_y_max;
  if (blocks_total_max <= 0) die("invalid block geometry");

  size_t a_bytes = (size_t)cfg.m * (size_t)cfg.k * sizeof(float);
  size_t b_bytes = (size_t)cfg.k * (size_t)cfg.n * sizeof(float);
  size_t c_bytes = (size_t)cfg.m * (size_t)cfg.n * sizeof(float);

  std::string out_dir = resolve_out_dir_from_exe();
  std::string iter_csv = out_dir + "/" + cfg.csv_prefix + "_iterations.csv";
  std::string summary_csv = out_dir + "/" + cfg.csv_prefix + "_summary.csv";

  printf("Device: %d SMs, %u TPCs (~%d SM/TPC)\n", num_sms, num_tpcs, sms_per_tpc);
  printf(
      "Dynamic single-process run: max_slices=%d warmup=%d hold_iters=%d sample_every=%d slice_rows=%d "
      "max_inflight=%d\n",
      cfg.max_slices, cfg.warmup, cfg.hold_iters, cfg.sample_every, cfg.slice_rows, cfg.max_inflight);
  printf("SM-range schedule length: %zu\n", schedule.size());
  printf("Matmul shape: M=%d N=%d K=%d, grid_x=%d (block=16x16)\n", cfg.m, cfg.n, cfg.k, blocks_x);

  // 3) Allocate reusable device buffers once to keep loop timing focused on
  // mask switching and launch cadence.
  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;
  cuda_check(cudaMalloc(&d_a, a_bytes), "cudaMalloc d_a");
  cuda_check(cudaMalloc(&d_b, b_bytes), "cudaMalloc d_b");
  cuda_check(cudaMalloc(&d_c, c_bytes), "cudaMalloc d_c");
  cuda_check(cudaMemset(d_a, 0x3f, a_bytes), "cudaMemset d_a");
  cuda_check(cudaMemset(d_b, 0x3f, b_bytes), "cudaMemset d_b");
  cuda_check(cudaMemset(d_c, 0, c_bytes), "cudaMemset d_c");

  dim3 block(16, 16);
  // 4) Warmup: execute a few tiny slices with stream creation/destruction.
  // This stabilizes CUDA runtime paths without polluting measured slice data. A matching prime
  // pass (below) then covers cudaMalloc(SMID) + cudaEvent used on every measured slice.
  for (int i = 0; i < cfg.warmup; i++) {
    int schedule_idx = (i / cfg.hold_iters) % (int)schedule.size();
    cudaStream_t s;
    cuda_check(cudaStreamCreate(&s), "cudaStreamCreate warmup");
    libsmctrl_set_stream_mask_ext((void*)s, schedule[schedule_idx].disable_mask);
    int m_slice = std::min(cfg.slice_rows, cfg.m);
    int blocks_y = (m_slice + 15) / 16;
    dim3 grid(blocks_x, blocks_y);
    tiled_matmul_slice_with_smid<<<grid, block, 0, s>>>(d_a, d_b, d_c, nullptr, 0, m_slice, cfg.m, cfg.n, cfg.k);
    cuda_check(cudaGetLastError(), "warmup launch");
    cuda_check(cudaStreamSynchronize(s), "warmup sync");
    cuda_check(cudaStreamDestroy(s), "cudaStreamDestroy warmup");
  }

  // Prime paths exercised on every measured slice but skipped above: optional SMID buffer
  // allocation (when sampling) and cudaEvent create/record/destroy. Without this, the first
  // logged slice often shows a large set_plus_launch_us spike even after warmup.
  {
    cudaStream_t s;
    cuda_check(cudaStreamCreate(&s), "cudaStreamCreate prime");
    const SmRange& req = schedule[0];
    libsmctrl_set_stream_mask_ext((void*)s, req.disable_mask);
    int m_slice = std::min(cfg.slice_rows, cfg.m);
    int blocks_y = (m_slice + 15) / 16;
    dim3 grid(blocks_x, blocks_y);
    int blocks_total = blocks_x * blocks_y;
    int* d_smid = nullptr;
    if (blocks_total > 0) {
      size_t smid_bytes = (size_t)blocks_total * sizeof(int);
      cuda_check(cudaMalloc(&d_smid, smid_bytes), "cudaMalloc prime d_smid");
    }
    tiled_matmul_slice_with_smid<<<grid, block, 0, s>>>(d_a, d_b, d_c, d_smid, 0, m_slice, cfg.m, cfg.n,
                                                          cfg.k);
    cuda_check(cudaGetLastError(), "prime launch");
    cudaEvent_t ev;
    cuda_check(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "prime cudaEventCreateWithFlags");
    cuda_check(cudaEventRecord(ev, s), "prime cudaEventRecord");
    cuda_check(cudaStreamSynchronize(s), "prime cudaStreamSynchronize");
    cuda_check(cudaEventDestroy(ev), "prime cudaEventDestroy");
    if (d_smid) cuda_check(cudaFree(d_smid), "cudaFree prime d_smid");
    cuda_check(cudaStreamDestroy(s), "cudaStreamDestroy prime");
  }

  std::vector<IterRow> rows;
  rows.reserve((size_t)std::min(cfg.max_slices, (cfg.m + cfg.slice_rows - 1) / cfg.slice_rows));
  uint64_t run_t0 = now_ns();
  int sampled_iters = 0;

  // 5) Core measured loop:
  // Execute one large GEMM by slicing over output rows. For each deployment:
  //   - create a fresh stream
  //   - apply the next mask to that stream
  //   - enqueue the GEMM slice on that stream (no per-slice sync)
  //   - record a completion event and keep enqueueing until the full GEMM is covered
  //
  // When max_inflight is reached, the oldest completed slice is drained (sync+destroy),
  // which is the only intentional backpressure besides GPU scheduling.
  std::deque<InflightSlice> inflight;
  int row0 = 0;
  int deploy_idx = 0;
  while (row0 < cfg.m && deploy_idx < cfg.max_slices) {
    while ((int)inflight.size() >= cfg.max_inflight) {
      finish_inflight_slice(inflight.front(), num_sms, sms_per_tpc, num_tpcs, rows, &sampled_iters);
      inflight.pop_front();
    }

    int schedule_idx = (deploy_idx / cfg.hold_iters) % (int)schedule.size();
    const SmRange& req = schedule[schedule_idx];
    bool do_sample = (deploy_idx % cfg.sample_every == 0);

    int m_slice = std::min(cfg.slice_rows, cfg.m - row0);
    int blocks_y = (m_slice + 15) / 16;
    int blocks_total = blocks_x * blocks_y;
    dim3 grid(blocks_x, blocks_y);

    uint64_t t0 = now_ns();
    cudaStream_t s;
    cuda_check(cudaStreamCreate(&s), "cudaStreamCreate slice");
    uint64_t t_after_create = now_ns();

    libsmctrl_set_stream_mask_ext((void*)s, req.disable_mask);
    uint64_t t1 = now_ns();

    int* d_smid = nullptr;
    if (do_sample) {
      size_t smid_bytes = (size_t)blocks_total * sizeof(int);
      cuda_check(cudaMalloc(&d_smid, smid_bytes), "cudaMalloc per-slice smid buffer");
    }

    tiled_matmul_slice_with_smid<<<grid, block, 0, s>>>(d_a, d_b, d_c, d_smid, row0, m_slice, cfg.m, cfg.n, cfg.k);
    cuda_check(cudaGetLastError(), "tiled_matmul_slice_with_smid launch");
    uint64_t t2 = now_ns();

    cudaEvent_t ev;
    cuda_check(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreateWithFlags");
    cuda_check(cudaEventRecord(ev, s), "cudaEventRecord slice");

    InflightSlice w;
    w.stream = s;
    w.done = ev;
    w.deploy_idx = deploy_idx;
    w.schedule_idx = schedule_idx;
    w.expected_lo = req.lo;
    w.expected_hi = req.hi;
    w.row0 = row0;
    w.m_slice = m_slice;
    w.blocks_total = blocks_total;
    w.sampled = do_sample;
    w.d_smid = d_smid;
    w.switch_us = (double)(t1 - t_after_create) / 1e3;
    w.launch_us = (double)(t2 - t_after_create) / 1e3;
    w.set_plus_launch_us = (double)(t2 - t0) / 1e3;
    inflight.push_back(w);

    row0 += m_slice;
    deploy_idx++;
  }

  while (!inflight.empty()) {
    finish_inflight_slice(inflight.front(), num_sms, sms_per_tpc, num_tpcs, rows, &sampled_iters);
    inflight.pop_front();
  }
  cuda_check(cudaDeviceSynchronize(), "final device sync");
  uint64_t run_t1 = now_ns();

  // 6) Aggregate timing and residency metrics for summary CSV.
  std::vector<double> sw;
  std::vector<double> total;
  sw.reserve(rows.size());
  total.reserve(rows.size());
  int mismatch_count = 0;
  double outside_sum = 0.0;
  for (const auto& r : rows) {
    sw.push_back(r.switch_us);
    total.push_back(r.set_plus_launch_us);
    if (r.sampled) {
      mismatch_count += r.mismatch;
      outside_sum += r.outside_pct;
    }
  }

  std::sort(sw.begin(), sw.end());
  std::sort(total.begin(), total.end());
  auto mean_of = [](const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return v.empty() ? 0.0 : (s / (double)v.size());
  };

  double mean_sw = mean_of(sw);
  double med_sw = sw.empty() ? 0.0 : sw[percentile_index((int)sw.size(), 0.50)];
  double p95_sw = sw.empty() ? 0.0 : sw[percentile_index((int)sw.size(), 0.95)];
  double p99_sw = sw.empty() ? 0.0 : sw[percentile_index((int)sw.size(), 0.99)];
  double mean_total = mean_of(total);
  double med_total = total.empty() ? 0.0 : total[percentile_index((int)total.size(), 0.50)];
  double p95_total = total.empty() ? 0.0 : total[percentile_index((int)total.size(), 0.95)];
  double p99_total = total.empty() ? 0.0 : total[percentile_index((int)total.size(), 0.99)];

  double run_sec = (double)(run_t1 - run_t0) / 1e9;
  double switches_per_sec = (run_sec > 0.0) ? ((double)rows.size() / run_sec) : 0.0;
  double avg_outside_pct = (sampled_iters > 0) ? (outside_sum / (double)sampled_iters) : 0.0;

  // Rows are finalized in completion order (FIFO drain). Sort by deploy index for plots.
  std::sort(rows.begin(), rows.end(), [](const IterRow& a, const IterRow& b) { return a.iter < b.iter; });

  // 7) Persist machine-readable artifacts for plotting/compare runs.
  write_iterations_csv(iter_csv, rows);
  write_summary_csv(summary_csv, (int)rows.size(), cfg.warmup, cfg.hold_iters, cfg.sample_every,
                    cfg.slice_rows,
                    cfg.max_inflight,
                    cfg.m, cfg.n, cfg.k, (int)schedule.size(), sampled_iters,
                    switches_per_sec, mean_sw, med_sw, p95_sw, p99_sw,
                    mean_total, med_total, p95_total, p99_total, mismatch_count, avg_outside_pct);

  printf("Switch us: mean=%.3f median=%.3f p95=%.3f p99=%.3f\n", mean_sw, med_sw, p95_sw, p99_sw);
  printf("Set+launch us: mean=%.3f median=%.3f p95=%.3f p99=%.3f\n",
         mean_total, med_total, p95_total, p99_total);
  printf("Switches/sec: %.2f\n", switches_per_sec);
  printf("Sampled iterations: %d, mismatches: %d, avg_outside_pct: %.3f%%\n",
         sampled_iters, mismatch_count, avg_outside_pct);
  printf("Wrote: %s\n", iter_csv.c_str());
  printf("Wrote: %s\n", summary_csv.c_str());

  // 8) Explicit teardown for clean repeated benchmark runs.
  cuda_check(cudaFree(d_c), "cudaFree d_c");
  cuda_check(cudaFree(d_b), "cudaFree d_b");
  cuda_check(cudaFree(d_a), "cudaFree d_a");
  return 0;
}
