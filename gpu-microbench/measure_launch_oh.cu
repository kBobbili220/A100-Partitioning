/* Copyright 2024 Joshua Bakita
 * Simple kernel that clocks how long a kernel launch takes to the GPU.
 * Prints samples to stdout and summary statistics to stderr.
 *
 * For best results, pin to a CPU core and redirect interrupts.
 */
#include <time.h>
#include <sched.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>

#include "testbench.h"

__global__ void flag_on_gpu(volatile int *flag) {
	*flag = 1;
}

// Get launch overhead
long measure_launch_overhead(bool first_run) {
	volatile int *barrier;
	struct timespec start, end;
	SAFE(cudaHostAlloc(&barrier, sizeof(*barrier), cudaHostAllocMapped));
	*barrier = 0;
	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	flag_on_gpu<<<100,100>>>(barrier);
	while (!*barrier) continue;
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	SAFE(cudaDeviceSynchronize());
	if (first_run)
		fprintf(stderr, "(%d) First launch completed at CLOCK_MONOTONIC_RAW = %ld ns\n",
				getpid(), time2ns(end));
	return timediff(start, end);
}

int main(int argc, char **argv) {
	int *__unused, i;
	unsigned long num_iters;
	long time;
	long double cumulative_time = 0;

	if (argc != 2 || !strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")) {
		fprintf(stderr, "Usage: %s [# of samples]\n", argv[0]);
		return 1;
	}

	num_iters = strtoul(argv[1], NULL, 10);

	// Initialize CUDA and a context (hack)
	SAFE(cudaMalloc(&__unused, 8));

	// Run once to ensure the kernel is compiled
	time = measure_launch_overhead(true);
	fprintf(stderr, "(%d) %ld ns (%.2f ms) warmup launch overhead\n", getpid(),
	        time, ns2ms((double)time));

	// Time several kernel launches
	for (i = 0; i < num_iters; i++) {
		// We're measuring on the order of microseconds. Make sure the
		// kernel scheduler does not interrupt us during a timing
		// iteration by explicitly invoking it here.
		sched_yield();
		// Measure overhead
		time = measure_launch_overhead(false);
		cumulative_time += time;
		// Print one sample per line
		fprintf(stdout, "%ld\n", time);
	}

	fprintf(stderr, "(%d) %.0Lf ns (%.2Lf ms) average launch overhead\n", getpid(),
	        cumulative_time / num_iters, ns2ms(cumulative_time / num_iters));
	return 0;
}
