/* Copyright 2024 Joshua Bakita
 * Tool that tracks preemptions as discontinuities in GPU time and logs them.
 *
 * The logged intervals are output to standard out as a CSV, with each row
 * containing a single interval. The first column contains the interval start
 * time in nanoseconds (ns), and the second column contains the interval end
 * time in ns.
 *
 * All progress and error messages and printed to standard error.
 *
 * Assumptions:
 * - The GPU clock continues to tick, regardless of if this is the current
 *   application running.
 */
#include <stdio.h>
#include <stdint.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "task_host_utilities.cu"
#include "testbench.h"

struct interval {
	uint64_t start;
	uint64_t end;
};

// Minimum time discontinuity which indicates a gap between intervals
// Clock resolution is about 1ns, but it only ticks every 1us pre-H100
#define MIN_PREEMPT_TICKS 2*1000 // ~2us

// Watch for discontinuities in the GPU clock, indicating intervals during
// which we we were preempted and another context ran.
// @param ivls     Base address of array to log interval start and end times
// @param num_ivls How many intervals should be logged before exiting?
__global__ void loop_on_gpu(struct interval *ivls, int num_ivls) {
	struct interval *curr_ivl = ivls;
	uint64_t last_time = GlobalTimer64();
	if (num_ivls <= 0)
		return;
	curr_ivl->start = last_time;
	// While no preemptions are occuring, this repeatedly obtains the clock,
	// storing the value in `last_time`. When we're switched away from and back
	// to, there will appear to be a large gap between `now` and `last_time`.
	// We record `last_time` as the end of the last interval, and `now` as the
	// start of the next interval, then resume repeatedly obtaining the clock.

	// Continue to log intervals until the pointer `curr_ivl` points to a
	// location off the end of the `ivls` array.
	while (curr_ivl < ivls + num_ivls) {
		uint64_t now = GlobalTimer64();
		if (now > last_time + MIN_PREEMPT_TICKS) {
			curr_ivl->end = last_time;
			curr_ivl++;
			curr_ivl->start = now;
		}
		last_time = now;
	}
}

static const char* usage_msg = "\
Usage: %s NUM_INTERVALS [-r]\n\
Spin on the GPU, logging intervals to stdout during which we are scheduled.\n\
  NUM_INTERVALS   Number of intervals of execution to log (logs one less than\n\
                  this number of preemptions).\n\
  -r, --raw       Print raw logged GPU times (skip conversion to CPU time).\n";

int main(int argc, char **argv) {
	struct interval *ivls_gpu, *ivls;
	struct timespec start, end, end_ivls_only;
	int num_ivls, skip_conversion;
	uint64_t dev_ns = 0;
	double d2h_scale = 1.0, host_s = 0;
	pid_t pid = getpid();

	// Argument parsing (could be improved)
	if (argc != 2 && argc != 3) {
		fprintf(stderr, usage_msg, argv[0]);
		return 1;
	}
	num_ivls = atoi(argv[1]);
	if (!num_ivls) {
		fprintf(stderr, usage_msg, argv[0]);
		return 1;
	}
	skip_conversion = argc == 3 &&
	    (!strcmp(argv[2], "-r") || !strcmp(argv[2], "--raw"));

	// Synchronize the GPU and CPU clocks (if requested) using the utilities
	// from task_host_utilities.cu.
	if (!skip_conversion) {
		// The skew is between -13 and 60 microseconds per second on the GTX
		// 1080, GTX 1060 3 GiB, GTX 970, RTX 6000 Ada, and GTX 1080 Ti.
		d2h_scale = InternalGetGPUTimerScale(0);
		InternalReadGPUNanoseconds(0, &host_s, &dev_ns);
		if (d2h_scale == -1 || (host_s == 0 && !dev_ns)) {
			fprintf(stderr, "Unable to synchronize time with the GPU. Aborting...\n");
			return 1;
		}
	}

	// Initialize GPU and CPU memory to store the intervals of execution
	// Note that we use GPU memory rather than pinned host memory in order to make
	// the time-monitoring loop on-GPU as fast as possible. (Pinned host memory is
	// slower to read/write from/to.)
	SAFE(cudaMalloc(&ivls_gpu, num_ivls * sizeof(struct interval)));
	ivls = (struct interval*)malloc(num_ivls * sizeof(struct interval));
	if (!ivls) {
		perror("While allocating memory");
		return 1;
	}

	// Monitor execution intervals
	fprintf(stderr, "(%d) Started monitoring execution for %d intervals...\n",
	    pid, num_ivls);
	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	loop_on_gpu<<<1,1>>>(ivls_gpu, num_ivls);
	SAFE(cudaDeviceSynchronize());
	clock_gettime(CLOCK_MONOTONIC_RAW, &end_ivls_only);

	// Pull back results onto the CPU
	SAFE(cudaMemcpy(ivls, ivls_gpu, num_ivls * sizeof(struct interval),
	    cudaMemcpyDeviceToHost));
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);

	// Print out the intervals, the start and end times of our monitor, and some
	// summary statistics
	for (int i = 0; i < num_ivls; i++) {
		if (skip_conversion) {
			printf("%lu, %lu\n", ivls[i].start, ivls[i].end);
		} else {
			// Convert all the interval times into CPU time.
			// First compute how far this interval's starting point is, in CPU seconds,
			// from the point where the device and CPU clocks were synchronized
			uint64_t start_time_since_sync = (ivls[i].start - dev_ns) * d2h_scale;
			// Then convert into absolute CPU time via offset adjustment
			uint64_t start = s2ns(host_s) + start_time_since_sync;
			// Repeat the same process for interval end times
			uint64_t end_time_since_sync = (ivls[i].end - dev_ns) * d2h_scale;
			uint64_t end = s2ns(host_s) + end_time_since_sync;
			printf("%lu, %lu\n", start, end);
		}
	}
	fprintf(stderr, "(%d) Total monitor runtime: %ld ms\n", pid,
	    ns2ms(timediff(start, end)));
	fprintf(stderr, "(%d) CPU logging start:     %ld ns\n", pid, time2ns(start));
	fprintf(stderr, "(%d) CPU logging end:       %ld ns\n", pid, time2ns(end));
	fprintf(stderr, "(%d) CPU copy out end:      %ld ns\n", pid, time2ns(end));
	if (!skip_conversion) {
		// Difference between CPU time of the start of the first interval and launch time
		int64_t launch_oh = s2ns(host_s) +
		    (ivls[0].start - dev_ns) * d2h_scale - time2ns(start);
		fprintf(stderr, "(%d) Aprox launch overhead: %ld ns\n", pid, launch_oh);
		fprintf(stderr, "(%d) CPU clock - GPU clock: %ld tick gap\n", pid,
		    (long)s2ns(host_s) - dev_ns);
		fprintf(stderr, "(%d) After 1 second, the GPU clock is %.f us %s (%.9fx d2h)\n",
		    pid, fabs((d2h_scale - 1) * 1e6),
		    d2h_scale > 1 ? "behind" : "ahead", d2h_scale);
	}

	return 0;
}
