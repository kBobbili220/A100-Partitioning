#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <limits.h>
#include <string>
#include <vector>
#include <unistd.h>

#include <cuda_runtime.h>

#include "libsmctrl.h"

// Top-level runtime configuration.
//
// This benchmark measures how quickly we can repeatedly update a stream-scoped
// SM/TPC mask while still verifying that each mask is actually respected by GPU
// scheduling. The knobs here intentionally separate:
// - schedule shape (`tpcs_spec`)
// - number of measured switch attempts (`iters`)
// - kernel footprint (`blocks`)
// - warmup stabilization (`warmup`)
// - output naming (`csv_prefix`)
struct Config {
  std::string tpcs_spec;
  int iters = 200;
  int blocks = 4096;
  int warmup = 20;
  std::string csv_prefix = "mask_switch_rate";
};

// Per-iteration record written to the "iterations" CSV.
//
// Timing fields:
// - switch_us: host-side time spent in libsmctrl_set_stream_mask_ext()
// - cycle_us: end-to-end iteration (set mask + launch + stream sync)
//
// Correctness fields:
// - requested_tpc: TPC we asked to enable (single-enabled-TPC mode)
// - dominant_tpc: most frequently observed TPC for this iteration's blocks
// - inside/outside/invalid breakdown based on SMID->TPC inference
// - mismatch flag for quick counting and plotting
struct IterRow {
  int iter = 0;
  int requested_tpc = -1;
  int dominant_tpc = -1;
  double switch_us = 0.0;
  double cycle_us = 0.0;
  int inside_blocks = 0;
  int outside_blocks = 0;
  int invalid_blocks = 0;
  double inside_pct = 0.0;
  double outside_pct = 0.0;
  int mismatch = 0;
};

// Hard fail helper for non-CUDA errors.
static void die(const char* msg) {
  fprintf(stderr, "fatal: %s\n", msg);
  std::exit(1);
}

// CUDA call guard; aborts immediately on error to keep benchmark traces clean.
static void cuda_check(cudaError_t e, const char* expr) {
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s (%d) at %s\n", cudaGetErrorString(e), (int)e, expr);
    std::exit(1);
  }
}

// Monotonic nanosecond timestamp for host-side measurements.
// CLOCK_MONOTONIC_RAW avoids wall-clock adjustments.
static uint64_t now_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

// Strict integer parser used by CLI argument decoding.
static bool parse_int(const char* s, int* out) {
  errno = 0;
  char* end = nullptr;
  long v = strtol(s, &end, 10);
  if (errno != 0 || !end || *end != '\0') return false;
  if (v < INT32_MIN || v > INT32_MAX) return false;
  *out = (int)v;
  return true;
}

// CLI help text.
static void usage(const char* argv0) {
  fprintf(stderr,
          "Usage: %s [--tpcs list] [--iters N] [--blocks N] [--warmup N] [--csv-prefix name]\n"
          "  --tpcs list/range of TPCs (default: all), e.g. 0,1,4-7\n",
          argv0);
}

// Parse and validate command-line options.
// We keep this explicit/manual so default behavior stays obvious and easy to
// tweak for microbenchmark experiments.
static Config parse_cli(int argc, char** argv) {
  Config c;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto need = [&](const char* f) -> const char* {
      if (i + 1 >= argc) {
        fprintf(stderr, "missing value for %s\n", f);
        std::exit(1);
      }
      i++;
      return argv[i];
    };
    if (a == "--help" || a == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else if (a == "--tpcs") {
      c.tpcs_spec = need("--tpcs");
    } else if (a == "--iters") {
      if (!parse_int(need("--iters"), &c.iters)) die("invalid --iters");
    } else if (a == "--blocks") {
      if (!parse_int(need("--blocks"), &c.blocks)) die("invalid --blocks");
    } else if (a == "--warmup") {
      if (!parse_int(need("--warmup"), &c.warmup)) die("invalid --warmup");
    } else if (a == "--csv-prefix") {
      c.csv_prefix = need("--csv-prefix");
    } else {
      usage(argv[0]);
      die("unknown argument");
    }
  }
  if (c.iters <= 0 || c.blocks <= 0 || c.warmup < 0) die("invalid numeric CLI options");
  return c;
}

// Return parent directory for a given absolute/relative path.
static std::string dirname_of_path(const std::string& path) {
  size_t slash = path.find_last_of('/');
  if (slash == std::string::npos) return ".";
  if (slash == 0) return "/";
  return path.substr(0, slash);
}

// Resolve benchmark output directory as "<directory of this executable>".
// This ensures CSV artifacts land in tests/mask_switch_rate regardless of CWD.
static std::string resolve_out_dir_from_exe() {
  char exe_buf[PATH_MAX];
  ssize_t n = readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
  if (n <= 0) die("failed to resolve /proc/self/exe");
  exe_buf[n] = '\0';
  return dirname_of_path(std::string(exe_buf));
}

// Parse a TPC list/range specification (e.g., "0,1,4-7"), or if empty return
// all available TPCs.
// Duplicates are removed via bitmap accumulation.
static std::vector<int> parse_tpc_set_or_all(const std::string& spec, uint32_t num_tpcs) {
  if (num_tpcs == 0) die("GPU reports zero TPCs");
  if (spec.empty()) {
    std::vector<int> all;
    all.reserve(num_tpcs);
    for (uint32_t i = 0; i < num_tpcs; i++) all.push_back((int)i);
    return all;
  }
  std::vector<int> mark(num_tpcs, 0);
  size_t pos = 0;
  while (pos < spec.size()) {
    size_t next = spec.find(',', pos);
    std::string tok = spec.substr(pos, (next == std::string::npos) ? std::string::npos : (next - pos));
    if (tok.empty()) die("empty token in --tpcs");
    size_t dash = tok.find('-');
    if (dash == std::string::npos) {
      int id = -1;
      if (!parse_int(tok.c_str(), &id)) die("invalid tpc token");
      if (id < 0 || id >= (int)num_tpcs) die("tpc token out of range");
      mark[id] = 1;
    } else {
      int lo = -1, hi = -1;
      std::string a = tok.substr(0, dash);
      std::string b = tok.substr(dash + 1);
      if (!parse_int(a.c_str(), &lo) || !parse_int(b.c_str(), &hi)) die("invalid tpc range");
      if (lo > hi) std::swap(lo, hi);
      if (lo < 0 || hi >= (int)num_tpcs) die("tpc range out of range");
      for (int t = lo; t <= hi; t++) mark[t] = 1;
    }
    if (next == std::string::npos) break;
    pos = next + 1;
  }
  std::vector<int> out;
  for (uint32_t i = 0; i < num_tpcs; i++) if (mark[i]) out.push_back((int)i);
  if (out.empty()) die("no TPCs selected");
  return out;
}

// libsmctrl mask semantics: bit=1 means "disabled".
// For this benchmark we enable exactly one TPC, so build a one-bit allow mask
// and invert it.
static uint128_t disable_mask_one_tpc(int tpc) {
  uint128_t en = ((uint128_t)1 << tpc);
  return ~en;
}

// Convert percentile in [0,1] to sorted-array index.
static int percentile_index(int n, double p) {
  if (n <= 0) return 0;
  int idx = (int)(p * (double)(n - 1));
  if (idx < 0) idx = 0;
  if (idx >= n) idx = n - 1;
  return idx;
}

// Safe percentage helper.
static double pct(int a, int total) {
  if (total <= 0) return 0.0;
  return 100.0 * (double)a / (double)total;
}

// Minimal probe kernel for fast "set mask -> launch -> verify placement" cycles.
// Thread 0 per block records SMID for host-side TPC inference.
__global__ void probe_kernel(float* out, uint16_t* smid_per_block) {
  int smid = 0;
  asm("mov.u32 %0, %%smid;" : "=r"(smid));
  if (threadIdx.x == 0) smid_per_block[blockIdx.x] = (uint16_t)smid;
  if (threadIdx.x == 0) out[blockIdx.x] = (float)smid;
}

// Infer TPC from SM ID using uniform "sms_per_tpc" partitioning.
// This is a practical heuristic for verification and mirrors other demos here.
static int tpc_of_sm(int sm, int sms_per_tpc, uint32_t num_tpcs) {
  if (sm < 0 || sms_per_tpc <= 0) return -1;
  int tpc = sm / sms_per_tpc;
  if (tpc < 0 || tpc >= (int)num_tpcs) return -1;
  return tpc;
}

// Per-iteration detail export.
static void write_iterations_csv(const std::string& path, const std::vector<IterRow>& rows) {
  FILE* f = fopen(path.c_str(), "w");
  if (!f) die("failed to open iterations csv");
  fprintf(f, "iter,requested_tpc,dominant_tpc,switch_us,cycle_us,inside_blocks,outside_blocks,invalid_blocks,inside_pct,outside_pct,mismatch\n");
  for (const auto& r : rows) {
    fprintf(f, "%d,%d,%d,%.6f,%.6f,%d,%d,%d,%.6f,%.6f,%d\n",
            r.iter, r.requested_tpc, r.dominant_tpc, r.switch_us, r.cycle_us, r.inside_blocks,
            r.outside_blocks, r.invalid_blocks, r.inside_pct, r.outside_pct, r.mismatch);
  }
  fclose(f);
}

// Aggregate summary export for quick parsing and plotting overlays.
static void write_summary_csv(const std::string& path,
                              const std::vector<IterRow>& rows,
                              double switches_per_sec,
                              double mean_sw, double med_sw, double p95_sw, double p99_sw,
                              double mean_cy, double med_cy, double p95_cy, double p99_cy,
                              int mismatch_count, double avg_outside_pct) {
  FILE* f = fopen(path.c_str(), "w");
  if (!f) die("failed to open summary csv");
  fprintf(f, "metric,value\n");
  fprintf(f, "iterations,%zu\n", rows.size());
  fprintf(f, "switches_per_sec,%.6f\n", switches_per_sec);
  fprintf(f, "switch_overhead_us_mean,%.6f\n", mean_sw);
  fprintf(f, "switch_overhead_us_median,%.6f\n", med_sw);
  fprintf(f, "switch_overhead_us_p95,%.6f\n", p95_sw);
  fprintf(f, "switch_overhead_us_p99,%.6f\n", p99_sw);
  fprintf(f, "cycle_us_mean,%.6f\n", mean_cy);
  fprintf(f, "cycle_us_median,%.6f\n", med_cy);
  fprintf(f, "cycle_us_p95,%.6f\n", p95_cy);
  fprintf(f, "cycle_us_p99,%.6f\n", p99_cy);
  fprintf(f, "mismatch_count,%d\n", mismatch_count);
  fprintf(f, "avg_outside_pct,%.6f\n", avg_outside_pct);
  fclose(f);
}

int main(int argc, char** argv) {
  // 1) Parse configuration and discover GPU topology.
  Config cfg = parse_cli(argc, argv);

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
  if (sms_per_tpc <= 0) die("invalid sms_per_tpc");

  // 2) Build round-robin switching schedule and output paths.
  std::vector<int> schedule = parse_tpc_set_or_all(cfg.tpcs_spec, num_tpcs);
  std::string out_dir = resolve_out_dir_from_exe();
  std::string iter_csv = out_dir + "/" + cfg.csv_prefix + "_iterations.csv";
  std::string summary_csv = out_dir + "/" + cfg.csv_prefix + "_summary.csv";

  printf("Device: %d SMs, %u TPCs (~%d SM/TPC)\n", num_sms, num_tpcs, sms_per_tpc);
  printf("Single-stream switch-rate run: iters=%d blocks=%d warmup=%d schedule_len=%zu\n",
         cfg.iters, cfg.blocks, cfg.warmup, schedule.size());

  // 3) Allocate one stream and reusable buffers.
  cudaStream_t stream;
  cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");
  float* d_out = nullptr;
  uint16_t* d_smids = nullptr;
  cuda_check(cudaMalloc(&d_out, sizeof(float) * cfg.blocks), "cudaMalloc d_out");
  cuda_check(cudaMalloc(&d_smids, sizeof(uint16_t) * cfg.blocks), "cudaMalloc d_smids");
  std::vector<uint16_t> h_smids(cfg.blocks);

  // 4) Warmup phase:
  //    Apply masks and run kernels without recording measurements to stabilize
  //    caches/launch state and reduce first-iteration skew.
  for (int i = 0; i < cfg.warmup; i++) {
    int req_tpc = schedule[i % schedule.size()];
    libsmctrl_set_stream_mask_ext((void*)stream, disable_mask_one_tpc(req_tpc));
    probe_kernel<<<cfg.blocks, 256, 0, stream>>>(d_out, d_smids);
  }
  cuda_check(cudaGetLastError(), "warmup launches");
  cuda_check(cudaStreamSynchronize(stream), "warmup sync");

  // 5) Measured switch loop:
  //    Each iteration sets a new mask, launches one kernel, synchronizes, then
  //    evaluates where the kernel actually ran.
  std::vector<IterRow> rows;
  rows.reserve(cfg.iters);
  uint64_t run_t0 = now_ns();
  for (int i = 0; i < cfg.iters; i++) {
    int req_tpc = schedule[i % schedule.size()];
    uint64_t sw0 = now_ns();
    libsmctrl_set_stream_mask_ext((void*)stream, disable_mask_one_tpc(req_tpc));
    uint64_t sw1 = now_ns();

    // cycle_us includes both software and GPU completion latency.
    uint64_t cy0 = sw0;
    probe_kernel<<<cfg.blocks, 256, 0, stream>>>(d_out, d_smids);
    cuda_check(cudaGetLastError(), "probe_kernel launch");
    cuda_check(cudaStreamSynchronize(stream), "iteration sync");
    uint64_t cy1 = now_ns();

    cuda_check(cudaMemcpy(h_smids.data(), d_smids, sizeof(uint16_t) * cfg.blocks, cudaMemcpyDeviceToHost),
               "smid memcpy");

    // Build inferred TPC histogram for this iteration.
    std::vector<int> tpc_hist(num_tpcs, 0);
    int inside = 0, outside = 0, invalid = 0;
    for (int b = 0; b < cfg.blocks; b++) {
      int sm = (int)h_smids[b];
      int tpc = tpc_of_sm(sm, sms_per_tpc, num_tpcs);
      if (tpc < 0) {
        invalid++;
        continue;
      }
      tpc_hist[tpc]++;
      if (tpc == req_tpc) inside++;
      else outside++;
    }

    // "dominant_tpc" is the strongest single-bin indicator of actual placement.
    int dominant_tpc = -1;
    int dominant_blocks = -1;
    for (uint32_t t = 0; t < num_tpcs; t++) {
      if (tpc_hist[t] > dominant_blocks) {
        dominant_blocks = tpc_hist[t];
        dominant_tpc = (int)t;
      }
    }

    IterRow r;
    r.iter = i;
    r.requested_tpc = req_tpc;
    r.dominant_tpc = dominant_tpc;
    r.switch_us = (double)(sw1 - sw0) / 1e3;
    r.cycle_us = (double)(cy1 - cy0) / 1e3;
    r.inside_blocks = inside;
    r.outside_blocks = outside;
    r.invalid_blocks = invalid;
    r.inside_pct = pct(inside, inside + outside);
    r.outside_pct = pct(outside, inside + outside);
    r.mismatch = (dominant_tpc != req_tpc) ? 1 : 0;
    rows.push_back(r);
  }
  uint64_t run_t1 = now_ns();

  // 6) Aggregate distribution statistics over all iterations.
  std::vector<double> sw, cy;
  sw.reserve(rows.size());
  cy.reserve(rows.size());
  int mismatch_count = 0;
  double outside_sum = 0.0;
  for (const auto& r : rows) {
    sw.push_back(r.switch_us);
    cy.push_back(r.cycle_us);
    mismatch_count += r.mismatch;
    outside_sum += r.outside_pct;
  }
  std::sort(sw.begin(), sw.end());
  std::sort(cy.begin(), cy.end());
  auto mean_of = [](const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return v.empty() ? 0.0 : (s / (double)v.size());
  };

  double mean_sw = mean_of(sw);
  double med_sw = sw.empty() ? 0.0 : sw[percentile_index((int)sw.size(), 0.50)];
  double p95_sw = sw.empty() ? 0.0 : sw[percentile_index((int)sw.size(), 0.95)];
  double p99_sw = sw.empty() ? 0.0 : sw[percentile_index((int)sw.size(), 0.99)];
  double mean_cy = mean_of(cy);
  double med_cy = cy.empty() ? 0.0 : cy[percentile_index((int)cy.size(), 0.50)];
  double p95_cy = cy.empty() ? 0.0 : cy[percentile_index((int)cy.size(), 0.95)];
  double p99_cy = cy.empty() ? 0.0 : cy[percentile_index((int)cy.size(), 0.99)];

  double run_sec = (double)(run_t1 - run_t0) / 1e9;
  double switches_per_sec = (run_sec > 0.0) ? ((double)rows.size() / run_sec) : 0.0;
  double avg_outside_pct = rows.empty() ? 0.0 : (outside_sum / (double)rows.size());

  // 7) Persist machine-readable results and print concise human summary.
  write_iterations_csv(iter_csv, rows);
  write_summary_csv(summary_csv, rows, switches_per_sec, mean_sw, med_sw, p95_sw, p99_sw,
                    mean_cy, med_cy, p95_cy, p99_cy, mismatch_count, avg_outside_pct);

  printf("Switch overhead us: mean=%.3f median=%.3f p95=%.3f p99=%.3f\n", mean_sw, med_sw, p95_sw, p99_sw);
  printf("Cycle time us:      mean=%.3f median=%.3f p95=%.3f p99=%.3f\n", mean_cy, med_cy, p95_cy, p99_cy);
  printf("Switches/sec: %.2f\n", switches_per_sec);
  printf("Correctness: mismatches=%d/%zu avg_outside_pct=%.3f%%\n",
         mismatch_count, rows.size(), avg_outside_pct);
  printf("Wrote: %s\n", iter_csv.c_str());
  printf("Wrote: %s\n", summary_csv.c_str());
  printf("Plot with: python3 ../tests/mask_switch_rate/plot_mask_switch_rate.py %s %s\n",
         iter_csv.c_str(), summary_csv.c_str());

  // 8) Explicit teardown.
  cuda_check(cudaFree(d_smids), "cudaFree d_smids");
  cuda_check(cudaFree(d_out), "cudaFree d_out");
  cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
  return 0;
}

