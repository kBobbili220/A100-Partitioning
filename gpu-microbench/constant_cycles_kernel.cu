/* Copyright 2021-2023 Joshua Bakita
 * Simple kernel that spins on the GPU for a specified number of iterations,
 * while tracking and printing the necessary CPU time.
 */
#include <time.h>
#include <stdio.h>
#include <cuda.h>

#include "testbench.h"

__global__ void loop_on_gpu(uint64_t iters, int *__unused) {
	for (volatile uint32_t i = 0; i < iters; i++) (*__unused)++;
}

int main(int argc, char **argv) {
	int res, *__unused;
	struct timespec start, end;

	if (argc != 2 || !strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")) {
		fprintf(stderr, "Usage: %s <# of millions of iterations, or -1 for infinite>\n",
		        argv[0]);
		return 1;
	}

	// Input is multiplied by one million, unless infinite
	uint64_t num_iters = strtoull(argv[1], NULL, 10);
	if (num_iters != (uint64_t)(-1))
		num_iters *= 1000 * 1000;

	// Initialize CUDA and a context (hack)
	SAFE(cudaMalloc(&__unused, 8));

	// Run iterations on a single thread
	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	loop_on_gpu<<<1,1>>>(num_iters, __unused);
	SAFE(cudaGetLastError() /* Check successful launch */);
	SAFE(cudaDeviceSynchronize());
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);

	// Print detailed timing information
	long elapsed = timediff(start, end);
	fprintf(stderr, "Started at %ld ns, ended at %ld ns\n",
	        s2ns(start.tv_sec) + start.tv_nsec, s2ns(end.tv_sec) + end.tv_nsec);
	fprintf(stderr, "%ld ns (%.2f ms) elapsed\n", elapsed, elapsed / (1000 * 1000.));

	// Verify success (fool optimizer)
	SAFE(cudaMemcpy(&res, __unused, 8, cudaMemcpyDeviceToHost));
	// Theoretically this can happen if `__unused` wraps around; maybe for a
	// very small `long` type, or a very long run. More likely indicates an error.
	if (!res)
		fprintf(stderr, "CRITICAL: Zero iterations seem completed. Likely incorrect "
		        "arguments, internal error, or corruption of CUDA internal state.\n");

	return 0;
}
