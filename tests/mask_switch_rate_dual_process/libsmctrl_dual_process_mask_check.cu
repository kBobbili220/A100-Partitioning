#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <unistd.h>
#include <vector>

#include <cuda_runtime.h>

#include "libsmctrl.h"

// Pure-speed mask+deploy microbenchmark.
//
// Goal:
// Measure host-side cadence of:
//   1) set SM mask via libsmctrl
//   2) launch kernel
//
// This version intentionally does NOT perform correctness checking and does NOT
// synchronize per iteration. It measures deployment speed only.

namespace {
// Number of measured iterations.
constexpr int kIters = 20000;
// Warmup iterations (not included in CSV/stats).
constexpr int kWarmup = 500;
// The only TPC we request in this benchmark.
constexpr int kTargetTpc = 0;
// Prefix for output artifacts.
constexpr const char* kPrefix = "dual_process_mask_check";
}  // namespace

// Per-iteration timing record written to iterations CSV.
struct IterRow {
  int iter = 0;
  double switch_us = 0.0;
  double launch_us = 0.0;
  double set_and_launch_us = 0.0;
  uint64_t iter_t0_ns = 0;
  uint64_t iter_t1_ns = 0;
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

// Monotonic host timer used for all benchmark timestamps.
static uint64_t now_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

// Basic path utility for choosing output directory.
static std::string dirname_of_path(const std::string& path) {
  size_t slash = path.find_last_of('/');
  if (slash == std::string::npos) return ".";
  if (slash == 0) return "/";
  return path.substr(0, slash);
}

// Write outputs next to executable so run location does not matter.
static std::string resolve_out_dir_from_exe() {
  char exe_buf[4096];
  ssize_t n = readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
  if (n <= 0) die("failed to resolve /proc/self/exe");
  exe_buf[n] = '\0';
  return dirname_of_path(std::string(exe_buf));
}

// libsmctrl semantics: bit=1 means disabled.
// This builds a mask that enables exactly one TPC and disables all others.
static uint128_t disable_mask_one_tpc(int tpc) {
  uint128_t en = ((uint128_t)1 << tpc);
  return ~en;
}

// Return index into sorted array for requested percentile.
static int percentile_index(int n, double p) {
  if (n <= 0) return 0;
  int idx = (int)(p * (double)(n - 1));
  if (idx < 0) idx = 0;
  if (idx >= n) idx = n - 1;
  return idx;
}

// Tiny kernel used only to create an actual launch event.
__global__ void probe_kernel() {}

// Write raw per-iteration timing data for plotting.
static void write_iterations_csv(const std::string& path, const std::vector<IterRow>& rows) {
  FILE* f = fopen(path.c_str(), "w");
  if (!f) die("failed to open iterations csv");
  fprintf(f, "iter,switch_us,launch_us,set_and_launch_us,iter_t0_ns,iter_t1_ns\n");
  for (const auto& r : rows) {
    fprintf(f, "%d,%.6f,%.6f,%.6f,%llu,%llu\n",
            r.iter, r.switch_us, r.launch_us, r.set_and_launch_us,
            (unsigned long long)r.iter_t0_ns, (unsigned long long)r.iter_t1_ns);
  }
  fclose(f);
}

// Write aggregate statistics for quick numeric summaries.
static void write_summary_csv(const std::string& path,
                              int total,
                              double mean_sw,
                              double med_sw,
                              double p95_sw,
                              double p99_sw,
                              double mean_launch,
                              double med_launch,
                              double p95_launch,
                              double p99_launch,
                              double mean_total,
                              double med_total,
                              double p95_total,
                              double p99_total,
                              double launches_per_sec) {
  FILE* f = fopen(path.c_str(), "w");
  if (!f) die("failed to open summary csv");
  fprintf(f, "metric,value\n");
  fprintf(f, "iterations,%d\n", total);
  fprintf(f, "switch_us_mean,%.6f\n", mean_sw);
  fprintf(f, "switch_us_median,%.6f\n", med_sw);
  fprintf(f, "switch_us_p95,%.6f\n", p95_sw);
  fprintf(f, "switch_us_p99,%.6f\n", p99_sw);
  fprintf(f, "launch_us_mean,%.6f\n", mean_launch);
  fprintf(f, "launch_us_median,%.6f\n", med_launch);
  fprintf(f, "launch_us_p95,%.6f\n", p95_launch);
  fprintf(f, "launch_us_p99,%.6f\n", p99_launch);
  fprintf(f, "set_and_launch_us_mean,%.6f\n", mean_total);
  fprintf(f, "set_and_launch_us_median,%.6f\n", med_total);
  fprintf(f, "set_and_launch_us_p95,%.6f\n", p95_total);
  fprintf(f, "set_and_launch_us_p99,%.6f\n", p99_total);
  fprintf(f, "launches_per_sec,%.6f\n", launches_per_sec);
  fclose(f);
}

int main() {
  // Sanity-check that selected TPC exists on this device.
  int num_sms = 0;
  cuda_check(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0),
             "cudaDeviceGetAttribute(MP count)");
  uint32_t num_tpcs = 0;
  int res = libsmctrl_get_tpc_info_cuda(&num_tpcs, 0);
  if (res != 0) die("libsmctrl_get_tpc_info_cuda failed");
  if (kTargetTpc < 0 || kTargetTpc >= (int)num_tpcs) die("target TPC out of range");

  cudaStream_t stream;
  cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");

  // Warmup: pre-touch runtime/driver paths so measured loop excludes one-time
  // startup effects as much as possible.
  for (int i = 0; i < kWarmup; i++) {
    libsmctrl_set_stream_mask_ext((void*)stream, disable_mask_one_tpc(kTargetTpc));
    probe_kernel<<<1, 1, 0, stream>>>();
  }
  cuda_check(cudaGetLastError(), "warmup launch");
  cuda_check(cudaStreamSynchronize(stream), "warmup sync");

  std::vector<IterRow> rows;
  rows.reserve(kIters);
  uint64_t run_t0 = now_ns();

  // ---------------------- Core measurement loop ----------------------
  // Each iteration intentionally measures only host-side "set + deploy":
  //
  // t0: before setting the stream mask
  // t1: immediately after set_stream_mask returns
  // t2: immediately after kernel launch API returns (enqueue done)
  //
  // Important: there is NO per-iteration synchronize here.
  // That means launch_us / set_and_launch_us represent CPU-side control-path
  // cost, not GPU completion time. This is exactly the "how fast can we set a
  // mask and deploy another kernel" metric.
  //
  // Kernel work completion is deferred and drained once at the end by a single
  // cudaStreamSynchronize().
  // ------------------------------------------------------------------
  for (int i = 0; i < kIters; i++) {
    uint64_t t0 = now_ns();
    libsmctrl_set_stream_mask_ext((void*)stream, disable_mask_one_tpc(kTargetTpc));
    uint64_t t1 = now_ns();
    probe_kernel<<<1, 1, 0, stream>>>();
    cuda_check(cudaGetLastError(), "probe_kernel launch");
    cuda_check(cudaStreamSynchronize(stream), "iteration sync");
    uint64_t t2 = now_ns();

    IterRow r;
    r.iter = i;
    r.iter_t0_ns = t0;
    r.iter_t1_ns = t2;
    r.switch_us = (double)(t1 - t0) / 1e3;
    r.launch_us = (double)(t2 - t1) / 1e3;
    // Combined host-side cost for one "mask set + kernel deploy" cycle.
    r.set_and_launch_us = (double)(t2 - t0) / 1e3;
    rows.push_back(r);
  }
  uint64_t run_t1 = now_ns();

  // Final synchronization ensures all queued launches complete before exit.
  cuda_check(cudaStreamSynchronize(stream), "final sync");

  std::vector<double> sw, launch, total;
  sw.reserve(rows.size());
  launch.reserve(rows.size());
  total.reserve(rows.size());
  for (const auto& r : rows) {
    sw.push_back(r.switch_us);
    launch.push_back(r.launch_us);
    total.push_back(r.set_and_launch_us);
  }
  std::sort(sw.begin(), sw.end());
  std::sort(launch.begin(), launch.end());
  std::sort(total.begin(), total.end());
  auto mean_of = [](const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return v.empty() ? 0.0 : s / (double)v.size();
  };

  double mean_sw = mean_of(sw);
  double med_sw = sw.empty() ? 0.0 : sw[percentile_index((int)sw.size(), 0.50)];
  double p95_sw = sw.empty() ? 0.0 : sw[percentile_index((int)sw.size(), 0.95)];
  double p99_sw = sw.empty() ? 0.0 : sw[percentile_index((int)sw.size(), 0.99)];

  double mean_launch = mean_of(launch);
  double med_launch = launch.empty() ? 0.0 : launch[percentile_index((int)launch.size(), 0.50)];
  double p95_launch = launch.empty() ? 0.0 : launch[percentile_index((int)launch.size(), 0.95)];
  double p99_launch = launch.empty() ? 0.0 : launch[percentile_index((int)launch.size(), 0.99)];

  double mean_total = mean_of(total);
  double med_total = total.empty() ? 0.0 : total[percentile_index((int)total.size(), 0.50)];
  double p95_total = total.empty() ? 0.0 : total[percentile_index((int)total.size(), 0.95)];
  double p99_total = total.empty() ? 0.0 : total[percentile_index((int)total.size(), 0.99)];

  double run_sec = (double)(run_t1 - run_t0) / 1e9;
  // Throughput of deployment loop: iterations per measured wall-clock second.
  // This is not kernel-completion throughput; it is control-path cadence.
  double launches_per_sec = (run_sec > 0.0) ? ((double)rows.size() / run_sec) : 0.0;

  std::string out_dir = resolve_out_dir_from_exe();
  std::string iter_csv = out_dir + "/" + kPrefix + "_iterations.csv";
  std::string summary_csv = out_dir + "/" + kPrefix + "_summary.csv";
  write_iterations_csv(iter_csv, rows);
  write_summary_csv(summary_csv, (int)rows.size(),
                    mean_sw, med_sw, p95_sw, p99_sw,
                    mean_launch, med_launch, p95_launch, p99_launch,
                    mean_total, med_total, p95_total, p99_total,
                    launches_per_sec);

  printf("Iterations: %zu\n", rows.size());
  printf("Launches/sec: %.2f\n", launches_per_sec);
  printf("set_and_launch_us mean=%.3f median=%.3f p95=%.3f p99=%.3f\n",
         mean_total, med_total, p95_total, p99_total);
  printf("Wrote: %s\n", iter_csv.c_str());
  printf("Wrote: %s\n", summary_csv.c_str());
  printf("Plot with: python3 ../tests/mask_switch_rate_dual_process/plot_dual_process_mask_check.py %s %s\n",
         iter_csv.c_str(), summary_csv.c_str());

  cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
  return 0;
}
