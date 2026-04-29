// Two-process alternating SM-mask benchmark: fork before CUDA, then two OS
// processes swap complementary TPC-derived stream masks in lockstep (pthread
// barriers in shared memory) while each advances an independent sliced GEMM.

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits.h>
#include <pthread.h>
#include <new>
#include <string>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include <cuda_runtime.h>

#include "libsmctrl.h"

// User-facing runtime options.
// This benchmark runs two *separate processes* on one GPU and alternates
// which partition each process is allowed to use every phase.
struct Config {
  std::string partition0_spec;
  std::string partition1_spec;
  int max_phases = 100000;
  int warmup = 5;
  int slice_rows = 256;
  int m = 2048;
  int n = 2048;
  int k = 2048;
  int device = 0;
  int drain_before_swap = 0;
  std::string csv_prefix = "two_process_dynamic_mask_matmul";
};

// One requested SM range plus the corresponding libsmctrl disable mask.
// Note: libsmctrl uses "1 bit = disabled" semantics.
struct SmRange {
  int lo = 0;
  int hi = 0;
  uint128_t disable_mask = ~((uint128_t)0);
};

// Cross-process synchronization primitives living in MAP_SHARED memory.
// phase_start: both processes must arrive before applying next mask/launch.
// phase_end:   both processes must finish the phase before moving on.
struct SharedSync {
  pthread_barrier_t phase_start;
  pthread_barrier_t phase_end;
};

// Per-phase measurement record written to CSV for plotting.
struct IterRow {
  int iter = 0;
  int phase = 0;
  int role = 0;
  int mask_id = 0;
  int expected_sm_lo = -1;
  int expected_sm_hi = -1;
  double switch_us = 0.0;
  double launch_us = 0.0;
  double set_plus_launch_us = 0.0;
};

// Fatal helper for consistent hard-fail behavior.
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

// Monotonic host timer used for latency windows and throughput.
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
          "Usage: %s --partition0 lo-hi --partition1 lo-hi [options]\n"
          "  Two complementary SM partitions (TPC-derived masks) swap between process A (parent)\n"
          "  and process B (child) each phase. Fork happens before any CUDA init.\n"
          "Options:\n"
          "  --max-phases N     cap phase iterations (default 100000)\n"
          "  --warmup N         per-process warmup slices (default 5)\n"
          "  --slice-rows N     rows per slice (default 256)\n"
          "  --m N --n N --k N  GEMM shape (default 2048)\n"
          "  --device N         CUDA device (default 0)\n"
          "  --drain-before-swap  cudaStreamSynchronize after each slice before barrier\n"
          "  --csv-prefix name  output prefix (default two_process_dynamic_mask_matmul)\n",
          argv0);
}

// Parse CLI and enforce required knobs.
// We require two explicit partitions because this benchmark is about swapping
// ownership of two known regions between processes.
static Config parse_cli(int argc, char** argv) {
  Config c;
  bool have0 = false, have1 = false;
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
    } else if (a == "--partition0") {
      c.partition0_spec = need("--partition0");
      have0 = true;
    } else if (a == "--partition1") {
      c.partition1_spec = need("--partition1");
      have1 = true;
    } else if (a == "--max-phases") {
      if (!parse_int(need("--max-phases"), &c.max_phases)) die("invalid --max-phases");
    } else if (a == "--warmup") {
      if (!parse_int(need("--warmup"), &c.warmup)) die("invalid --warmup");
    } else if (a == "--slice-rows") {
      if (!parse_int(need("--slice-rows"), &c.slice_rows)) die("invalid --slice-rows");
    } else if (a == "--m") {
      if (!parse_int(need("--m"), &c.m)) die("invalid --m");
    } else if (a == "--n") {
      if (!parse_int(need("--n"), &c.n)) die("invalid --n");
    } else if (a == "--k") {
      if (!parse_int(need("--k"), &c.k)) die("invalid --k");
    } else if (a == "--device") {
      if (!parse_int(need("--device"), &c.device)) die("invalid --device");
    } else if (a == "--drain-before-swap") {
      c.drain_before_swap = 1;
    } else if (a == "--csv-prefix") {
      c.csv_prefix = need("--csv-prefix");
    } else {
      usage(argv[0]);
      die("unknown argument");
    }
  }
  if (!have0 || !have1) die("require --partition0 and --partition1");
  if (c.max_phases <= 0 || c.warmup < 0 || c.slice_rows <= 0) die("invalid numeric options");
  if (c.m <= 0 || c.n <= 0 || c.k <= 0) die("matrix dimensions must be positive");
  return c;
}

static std::string dirname_of_path(const std::string& path) {
  size_t slash = path.find_last_of('/');
  if (slash == std::string::npos) return ".";
  if (slash == 0) return "/";
  return path.substr(0, slash);
}

// Outputs are written next to the executable so run cwd does not matter.
static std::string resolve_out_dir_from_exe() {
  char exe_buf[PATH_MAX];
  ssize_t n = readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
  if (n <= 0) die("failed to resolve /proc/self/exe");
  exe_buf[n] = '\0';
  return dirname_of_path(std::string(exe_buf));
}

// Parse "lo-hi" SM range and convert it into a TPC disable mask.
// Mapping is approximate and topology-dependent: SM -> TPC by integer division.
static SmRange parse_one_partition(const std::string& tok, int num_sms, int sms_per_tpc, uint32_t num_tpcs) {
  size_t dash = tok.find('-');
  if (dash == std::string::npos) die("partition must be lo-hi");
  int lo = -1, hi = -1;
  if (!parse_int(tok.substr(0, dash).c_str(), &lo) || !parse_int(tok.substr(dash + 1).c_str(), &hi)) {
    die("invalid partition token");
  }
  if (lo > hi) std::swap(lo, hi);
  if (lo < 0 || hi >= num_sms) die("partition SM range out of bounds");
  uint128_t enable_mask = 0;
  for (int sm = lo; sm <= hi; sm++) {
    int tpc = sm / sms_per_tpc;
    if (tpc >= 0 && tpc < (int)num_tpcs) enable_mask |= ((uint128_t)1 << tpc);
  }
  SmRange r;
  r.lo = lo;
  r.hi = hi;
  r.disable_mask = ~enable_mask;
  return r;
}

// Extract only valid TPC enable bits from a disable-mask representation.
static uint128_t tpc_enable_bits(const SmRange& r, uint32_t num_tpcs) {
  uint128_t dm = r.disable_mask;
  uint128_t en = ~dm;
  if (num_tpcs < 128) {
    uint128_t mask = (((uint128_t)1 << num_tpcs) - 1);
    en &= mask;
  }
  return en;
}

// Enforce disjoint partitions *after* SM->TPC mapping.
// This avoids "logical SM ranges looked disjoint, but mapped TPC bits overlap".
static void validate_disjoint_partitions(const SmRange& p0, const SmRange& p1, uint32_t num_tpcs) {
  uint128_t e0 = tpc_enable_bits(p0, num_tpcs);
  uint128_t e1 = tpc_enable_bits(p1, num_tpcs);
  if ((e0 & e1) != 0) die("partitions overlap in TPC enable bits after SM→TPC mapping");
}

static int percentile_index(int n, double p) {
  if (n <= 0) return 0;
  int idx = (int)(p * (double)(n - 1));
  if (idx < 0) idx = 0;
  if (idx >= n) idx = n - 1;
  return idx;
}

// 16x16 tiled GEMM slice kernel.
// block_smid is optional (nullptr in this benchmark): left in signature to
// keep parity with single-process benchmark kernel shape.
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

  if (block_smid != nullptr && threadIdx.x == 0 && threadIdx.y == 0) {
    int smid = 0;
    asm("mov.u32 %0, %%smid;" : "=r"(smid));
    block_smid[blockIdx.y * gridDim.x + blockIdx.x] = smid;
  }

  for (int tile = 0; tile < (k + 15) / 16; tile++) {
    int a_col = tile * 16 + threadIdx.x;
    int b_row = tile * 16 + threadIdx.y;

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

// One shard per process to avoid concurrent file writes from both processes.
static void write_iterations_csv(const std::string& path, const std::vector<IterRow>& rows) {
  FILE* f = fopen(path.c_str(), "w");
  if (!f) die("failed to open iterations csv");
  fprintf(f,
          "iter,phase,role,mask_id,expected_sm_lo,expected_sm_hi,switch_us,launch_us,set_plus_launch_us\n");
  for (const auto& r : rows) {
    fprintf(f, "%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f\n", r.iter, r.phase, r.role, r.mask_id, r.expected_sm_lo,
            r.expected_sm_hi, r.switch_us, r.launch_us, r.set_plus_launch_us);
  }
  fclose(f);
}

// Merge A+B shards into one deterministic file sorted by (iter, role).
// Sorting is important for plotting and stable diffs across runs.
static void merge_iteration_files(const std::string& path_a,
                                  const std::string& path_b,
                                  const std::string& path_out) {
  std::vector<IterRow> all;
  auto load_shard = [&](const char* p) {
    FILE* in = fopen(p, "r");
    if (!in) die("failed to open iteration shard for merge");
    char line[4096];
    if (!fgets(line, sizeof(line), in)) {
      fclose(in);
      return;
    }
    while (fgets(line, sizeof(line), in)) {
      IterRow r{};
      int n = sscanf(line, "%d,%d,%d,%d,%d,%d,%lf,%lf,%lf", &r.iter, &r.phase, &r.role, &r.mask_id,
                     &r.expected_sm_lo, &r.expected_sm_hi, &r.switch_us, &r.launch_us, &r.set_plus_launch_us);
      if (n == 9) all.push_back(r);
    }
    fclose(in);
  };
  load_shard(path_a.c_str());
  load_shard(path_b.c_str());
  std::sort(all.begin(), all.end(), [](const IterRow& a, const IterRow& b) {
    if (a.iter != b.iter) return a.iter < b.iter;
    return a.role < b.role;
  });

  FILE* out = fopen(path_out.c_str(), "w");
  if (!out) die("failed to open merged iterations csv");
  fprintf(out,
          "iter,phase,role,mask_id,expected_sm_lo,expected_sm_hi,switch_us,launch_us,set_plus_launch_us\n");
  for (const auto& r : all) {
    fprintf(out, "%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f\n", r.iter, r.phase, r.role, r.mask_id, r.expected_sm_lo,
            r.expected_sm_hi, r.switch_us, r.launch_us, r.set_plus_launch_us);
  }
  fclose(out);
}

// Summary over merged rows. These are host-side deployment timings, not
// kernel completion times (unless drain-before-swap is enabled).
static void write_summary_from_rows(const std::string& path,
                                    const std::vector<IterRow>& all,
                                    const Config& cfg,
                                    int num_sms,
                                    uint32_t num_tpcs,
                                    double wall_sec) {
  std::vector<double> sw, tot;
  for (const auto& r : all) {
    sw.push_back(r.switch_us);
    tot.push_back(r.set_plus_launch_us);
  }
  std::sort(sw.begin(), sw.end());
  std::sort(tot.begin(), tot.end());
  auto mean_of = [](const std::vector<double>& v) {
    double s = 0;
    for (double x : v) s += x;
    return v.empty() ? 0.0 : s / (double)v.size();
  };
  int n = (int)sw.size();
  double mean_sw = mean_of(sw);
  double med_sw = n ? sw[percentile_index(n, 0.50)] : 0.0;
  double p95_sw = n ? sw[percentile_index(n, 0.95)] : 0.0;
  double p99_sw = n ? sw[percentile_index(n, 0.99)] : 0.0;
  double mean_tot = mean_of(tot);
  double med_tot = n ? tot[percentile_index(n, 0.50)] : 0.0;
  double p95_tot = n ? tot[percentile_index(n, 0.95)] : 0.0;
  double p99_tot = n ? tot[percentile_index(n, 0.99)] : 0.0;
  double phases_per_sec = (wall_sec > 0.0 && all.size() > 0) ? ((double)all.size() / wall_sec) : 0.0;

  FILE* f = fopen(path.c_str(), "w");
  if (!f) die("failed to open summary csv");
  fprintf(f, "metric,value\n");
  fprintf(f, "rows_merged,%zu\n", all.size());
  fprintf(f, "max_phases,%d\n", cfg.max_phases);
  fprintf(f, "warmup,%d\n", cfg.warmup);
  fprintf(f, "slice_rows,%d\n", cfg.slice_rows);
  fprintf(f, "matrix_m,%d\n", cfg.m);
  fprintf(f, "matrix_n,%d\n", cfg.n);
  fprintf(f, "matrix_k,%d\n", cfg.k);
  fprintf(f, "device,%d\n", cfg.device);
  fprintf(f, "drain_before_swap,%d\n", cfg.drain_before_swap);
  fprintf(f, "num_sms,%d\n", num_sms);
  fprintf(f, "num_tpcs,%u\n", num_tpcs);
  fprintf(f, "wall_run_sec,%.9f\n", wall_sec);
  fprintf(f, "merged_rows_per_sec,%.6f\n", phases_per_sec);
  fprintf(f, "switch_us_mean,%.6f\n", mean_sw);
  fprintf(f, "switch_us_median,%.6f\n", med_sw);
  fprintf(f, "switch_us_p95,%.6f\n", p95_sw);
  fprintf(f, "switch_us_p99,%.6f\n", p99_sw);
  fprintf(f, "set_plus_launch_us_mean,%.6f\n", mean_tot);
  fprintf(f, "set_plus_launch_us_median,%.6f\n", med_tot);
  fprintf(f, "set_plus_launch_us_p95,%.6f\n", p95_tot);
  fprintf(f, "set_plus_launch_us_p99,%.6f\n", p99_tot);
  fclose(f);
}

// Core swap policy:
// - even phase: role A gets P0, role B gets P1
// - odd  phase: role A gets P1, role B gets P0
static const SmRange& mask_for_role_phase(int role, int phase, const SmRange& p0, const SmRange& p1) {
  bool a_gets_p0 = (phase % 2) == 0;
  if (role == 0) {
    return a_gets_p0 ? p0 : p1;
  }
  return a_gets_p0 ? p1 : p0;
}

// Worker logic executed by both parent (role=0) and child (role=1).
// Important: each process initializes CUDA independently after fork().
static void run_role(int role, const Config& cfg, SharedSync* sync, const std::string& iter_path) {
  cuda_check(cudaSetDevice(cfg.device), "cudaSetDevice");

  int num_sms = 0;
  cuda_check(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, cfg.device),
             "cudaDeviceGetAttribute(MP count)");
  uint32_t num_tpcs = 0;
  int res = libsmctrl_get_tpc_info_cuda(&num_tpcs, cfg.device);
  if (res != 0) die("libsmctrl_get_tpc_info_cuda failed");
  int sms_per_tpc = (num_tpcs == 0) ? 0 : (num_sms / (int)num_tpcs);
  if (sms_per_tpc <= 0) die("invalid sms_per_tpc");

  // Rebuild partition masks per process so each process is self-contained.
  SmRange p0 = parse_one_partition(cfg.partition0_spec, num_sms, sms_per_tpc, num_tpcs);
  SmRange p1 = parse_one_partition(cfg.partition1_spec, num_sms, sms_per_tpc, num_tpcs);
  validate_disjoint_partitions(p0, p1, num_tpcs);

  int blocks_x = (cfg.n + 15) / 16;
  dim3 block(16, 16);

  size_t a_bytes = (size_t)cfg.m * (size_t)cfg.k * sizeof(float);
  size_t b_bytes = (size_t)cfg.k * (size_t)cfg.n * sizeof(float);
  size_t c_bytes = (size_t)cfg.m * (size_t)cfg.n * sizeof(float);

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;
  cuda_check(cudaMalloc(&d_a, a_bytes), "cudaMalloc d_a");
  cuda_check(cudaMalloc(&d_b, b_bytes), "cudaMalloc d_b");
  cuda_check(cudaMalloc(&d_c, c_bytes), "cudaMalloc d_c");
  cuda_check(cudaMemset(d_a, 0x3f, a_bytes), "cudaMemset d_a");
  cuda_check(cudaMemset(d_b, 0x3f, b_bytes), "cudaMemset d_b");
  cuda_check(cudaMemset(d_c, 0, c_bytes), "cudaMemset d_c");

  cudaStream_t stream{};
  cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");

  // Warmup touches set-mask + launch + sync paths before measured rows.
  for (int w = 0; w < cfg.warmup; w++) {
    int ph = w & 1;
    const SmRange& req = mask_for_role_phase(role, ph, p0, p1);
    libsmctrl_set_stream_mask_ext((void*)stream, req.disable_mask);
    int m_slice = std::min(cfg.slice_rows, cfg.m);
    int blocks_y = (m_slice + 15) / 16;
    dim3 grid(blocks_x, blocks_y);
    tiled_matmul_slice_with_smid<<<grid, block, 0, stream>>>(d_a, d_b, d_c, nullptr, 0, m_slice, cfg.m, cfg.n,
                                                               cfg.k);
    cuda_check(cudaGetLastError(), "warmup launch");
    cuda_check(cudaStreamSynchronize(stream), "warmup sync");
  }

  std::vector<IterRow> rows;
  rows.reserve((size_t)cfg.max_phases + 1);

  int row0 = 0;
  // Each iteration is one "phase step" for this process.
  // row0 advances so each process independently completes its own GEMM.
  for (int iter = 0; row0 < cfg.m && iter < cfg.max_phases; iter++) {
    int phase = iter & 1;
    const SmRange& req = mask_for_role_phase(role, phase, p0, p1);
    int mask_id = (req.lo == p0.lo && req.hi == p0.hi) ? 0 : 1;

    // Rendezvous so both processes flip/apply for the same logical phase.
    pthread_barrier_wait(&sync->phase_start);

    // Timed host window:
    // t0 -> t1 : libsmctrl_set_stream_mask_ext only
    // t1 -> t2 : launch enqueue path only
    // t0 -> t2 : full set+launch deployment window
    uint64_t t0 = now_ns();
    libsmctrl_set_stream_mask_ext((void*)stream, req.disable_mask);
    uint64_t t1 = now_ns();

    int m_slice = std::min(cfg.slice_rows, cfg.m - row0);
    int blocks_y = (m_slice + 15) / 16;
    dim3 grid(blocks_x, blocks_y);
    tiled_matmul_slice_with_smid<<<grid, block, 0, stream>>>(d_a, d_b, d_c, nullptr, row0, m_slice, cfg.m, cfg.n,
                                                              cfg.k);
    cuda_check(cudaGetLastError(), "slice launch");
    uint64_t t2 = now_ns();

    // Optional "clean handoff" mode:
    // drain this process stream before allowing the next phase swap.
    if (cfg.drain_before_swap) cuda_check(cudaStreamSynchronize(stream), "drain cudaStreamSynchronize");

    // End-of-phase barrier prevents one process from racing ahead in phase id.
    pthread_barrier_wait(&sync->phase_end);

    IterRow r;
    r.iter = iter;
    r.phase = phase;
    r.role = role;
    r.mask_id = mask_id;
    r.expected_sm_lo = req.lo;
    r.expected_sm_hi = req.hi;
    r.switch_us = (double)(t1 - t0) / 1e3;
    r.launch_us = (double)(t2 - t1) / 1e3;
    r.set_plus_launch_us = (double)(t2 - t0) / 1e3;
    rows.push_back(r);

    row0 += m_slice;
  }

  cuda_check(cudaStreamSynchronize(stream), "final stream sync");
  cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
  cuda_check(cudaFree(d_a), "cudaFree d_a");
  cuda_check(cudaFree(d_b), "cudaFree d_b");
  cuda_check(cudaFree(d_c), "cudaFree d_c");

  write_iterations_csv(iter_path, rows);
}

int main(int argc, char** argv) {
  Config cfg = parse_cli(argc, argv);

  // Shared barrier state for both processes (parent + one child).
  size_t sync_sz = sizeof(SharedSync);
  if (sync_sz < (size_t)sysconf(_SC_PAGESIZE)) sync_sz = (size_t)sysconf(_SC_PAGESIZE);
  void* mem = mmap(nullptr, sync_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  if (mem == MAP_FAILED) die("mmap shared sync failed");
  SharedSync* sync = new (mem) SharedSync();

  pthread_barrierattr_t ba;
  if (pthread_barrierattr_init(&ba) != 0) die("pthread_barrierattr_init");
  if (pthread_barrierattr_setpshared(&ba, PTHREAD_PROCESS_SHARED) != 0) die("pthread_barrierattr_setpshared");
  if (pthread_barrier_init(&sync->phase_start, &ba, 2) != 0) die("pthread_barrier_init phase_start");
  if (pthread_barrier_init(&sync->phase_end, &ba, 2) != 0) die("pthread_barrier_init phase_end");
  pthread_barrierattr_destroy(&ba);

  std::string out_dir = resolve_out_dir_from_exe();
  std::string path_a = out_dir + "/" + cfg.csv_prefix + "_iterations_A.csv";
  std::string path_b = out_dir + "/" + cfg.csv_prefix + "_iterations_B.csv";
  std::string path_merged = out_dir + "/" + cfg.csv_prefix + "_iterations.csv";
  std::string path_summary = out_dir + "/" + cfg.csv_prefix + "_summary.csv";

  uint64_t wall0 = now_ns();
  // Critical design point: fork before any CUDA runtime usage in this process.
  pid_t pid = fork();
  if (pid < 0) die("fork failed");

  if (pid == 0) {
    // Child process = role B.
    run_role(1, cfg, sync, path_b);
    _exit(0);
  }

  // Parent process = role A.
  run_role(0, cfg, sync, path_a);

  int st = 0;
  if (waitpid(pid, &st, 0) < 0) die("waitpid failed");
  if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) {
    fprintf(stderr, "child exited abnormally (status=%d)\n", st);
    std::exit(1);
  }

  pthread_barrier_destroy(&sync->phase_start);
  pthread_barrier_destroy(&sync->phase_end);
  munmap(mem, sync_sz);

  uint64_t wall1 = now_ns();
  double wall_sec = (double)(wall1 - wall0) / 1e9;

  // Parent combines shards after both processes complete.
  merge_iteration_files(path_a, path_b, path_merged);

  std::vector<IterRow> merged;
  {
    FILE* f = fopen(path_merged.c_str(), "r");
    if (!f) die("failed to read merged csv");
    char line[4096];
    if (!fgets(line, sizeof(line), f)) {
      fclose(f);
      die("empty merged csv");
    }
    while (fgets(line, sizeof(line), f)) {
      IterRow r{};
      int n = sscanf(line, "%d,%d,%d,%d,%d,%d,%lf,%lf,%lf", &r.iter, &r.phase, &r.role, &r.mask_id,
                     &r.expected_sm_lo, &r.expected_sm_hi, &r.switch_us, &r.launch_us, &r.set_plus_launch_us);
      if (n == 9) merged.push_back(r);
    }
    fclose(f);
  }

  int num_sms = 0;
  cuda_check(cudaSetDevice(cfg.device), "cudaSetDevice summary");
  cuda_check(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, cfg.device), "cudaDeviceGetAttribute");
  uint32_t num_tpcs = 0;
  if (libsmctrl_get_tpc_info_cuda(&num_tpcs, cfg.device) != 0) num_tpcs = 0;

  // Summary is produced once from merged rows for easier downstream plotting.
  write_summary_from_rows(path_summary, merged, cfg, num_sms, num_tpcs, wall_sec);

  printf("Wrote: %s\n", path_merged.c_str());
  printf("Wrote: %s\n", path_summary.c_str());
  printf("Wall clock (both processes): %.6f s, merged rows: %zu\n", wall_sec, merged.size());
  return 0;
}
