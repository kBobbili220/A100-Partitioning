/* Copyright 2024 Joshua Bakita
 * Serially dispatch copies in one direction and record the transfer times
 * In order to both record individual copy times, and maintain constant
 * pending copies, this program relies on the implict synchronization which
 * occurs in CUDA when unidirectional copies are dispatched into separate
 * streams.
 *
 * Parameters include the size of the copy to perform and the number of times
 * to perform said copy.
 */
#include <stdio.h>
#include <cuda.h>

#include "copy_testbench.h"

typedef enum {
	MODE_TO_GPU,
	MODE_FROM_GPU,
	MODE_PEER,
} copy_mode_t;

void usage(char** argv) {
	fprintf(stderr, "Usage: %s <# of 4KiB pages to copy> <# of iterations> "
	       "<direction [to/from/peer] GPU>\n", argv[0]);
}

int main(int argc, char** argv) {
	cudaError_t err;
	int dev1, dev2;
	size_t i;
	char *pinned_hostmem, *devmem, *devmem_d2;
	cudaStream_t even_stream, odd_stream;
	struct timespec even_finish, odd_finish;
	copy_mode_t mode;

	if (argc != 4) {
		usage(argv);
		return 1;
	}

	// Size of each copy to the GPU, in bytes
	const size_t COPY_SIZE = strtoul(argv[1], NULL, 10) * PG_SZ;
	// Total copies = NUM_ITERS * 2 + 1
	const size_t NUM_ITERS = strtoul(argv[2], NULL, 10);
	if (strcmp(argv[3], "to") == 0)
		mode = MODE_TO_GPU;
	else if (strcmp(argv[3], "from") == 0)
		mode = MODE_FROM_GPU;
	else if (strcmp(argv[3], "peer") == 0)
		mode = MODE_PEER;
	else {
		usage(argv);
		return 1;
	}
	fprintf(stderr, "Copying %s %lu times %s.\n", human_readable_bytes(COPY_SIZE),
	        NUM_ITERS * 2 + 1, argv[3]);

	// When doing GPU-to-GPU copy, what is the source device?
	if (mode == MODE_PEER) {
		SAFE(cudaGetDevice(&dev1));
		if (dev1 == 0)
			dev2 = 1;
		else
			dev2 = 0;
	}

	SAFE(cudaMallocHost(&pinned_hostmem, COPY_SIZE));
	// Populate pinned_hostmem with random data
	for (i = 0; i < COPY_SIZE; i++)
		// Don't allow 0 so that the copy detection logic works
		pinned_hostmem[i] = max((rand() & 0xff), 1);
	SAFE(cudaMalloc(&devmem, COPY_SIZE));
	SAFE(cudaMemset(devmem, 0, COPY_SIZE));
	if (mode == MODE_PEER) {
		err = cudaSetDevice(dev2);
		if (err)
			fprintf(stderr, "Warning: No other device available. Treating 'peer' as an internal device transfer.\n");
		SAFE(cudaMalloc(&devmem_d2, COPY_SIZE));
		SAFE(cudaMemset(devmem_d2, 0, COPY_SIZE));
		SAFE(cudaSetDevice(dev1));
	}

	// If copying from device (to CPU or peer), first copy over initialization data
	if (mode != MODE_TO_GPU) {
		// Use a synchronizing memory copy
		SAFE(cudaMemcpy(devmem, pinned_hostmem, COPY_SIZE, cudaMemcpyHostToDevice));
	}

	// Initialize two streams for alternating dispatch and synchronization
	SAFE(cudaStreamCreate(&even_stream));
	SAFE(cudaStreamCreate(&odd_stream));

	// Reconfigure depending on the copy type
	void *target, *source;
	cudaMemcpyKind direction;
	switch (mode) {
		case MODE_TO_GPU:
			target = devmem;
			source = pinned_hostmem;
			direction = cudaMemcpyHostToDevice;
			break;
		case MODE_FROM_GPU:
			target = pinned_hostmem;
			source = devmem;
			direction = cudaMemcpyDeviceToHost;
			break;
		case MODE_PEER:
			target = devmem;
			source = devmem_d2;
			direction = cudaMemcpyDeviceToDevice;
			break;
		default:
			fprintf(stderr, "FATAL ERROR: 'mode' is not a valid enum value.\n");
			return 1;
	}

	// Kick off the first copy that doesn't include a log
	clock_gettime(CLOCK_MONOTONIC_RAW, &even_finish);
	SAFE(cudaMemcpyAsync(target, source, COPY_SIZE, direction, odd_stream));
	// Primary timing and dispatch loop. NUM_ITERS * 2 + 1 copies will be done
	for (i = 1; NUM_ITERS == (size_t) -1 || i < NUM_ITERS + 2;) {
		// Odd copy is underway. Queue up even copy so that there's no break.
		SAFE(cudaMemcpyAsync(target, source, COPY_SIZE, direction, even_stream));
		SAFE(cudaStreamSynchronize(odd_stream));
		// Odd copy has finished. Record and print time.
		clock_gettime(CLOCK_MONOTONIC_RAW, &odd_finish);
		// Time for the 'odd' copy: odd_finish (end) - even_finish (start)
		printf("%s/%lu: %ldms\n", argv[3], i++, ns2ms(timediff(even_finish, odd_finish)));
		// Even copy is underway. Queue up odd copy so that there's no break.
		SAFE(cudaMemcpyAsync(target, source, COPY_SIZE, direction, odd_stream));
		SAFE(cudaStreamSynchronize(even_stream));
		// Even copy has finished. Record and print time.
		clock_gettime(CLOCK_MONOTONIC_RAW, &even_finish);
		// Time for the 'even' copy: even_finish (end) - odd_finish (start)
		printf("%s/%lu: %ldms\n", argv[3], i++, ns2ms(timediff(odd_finish, even_finish)));
	}
	// Record the time for the last copy to end
	SAFE(cudaStreamSynchronize(odd_stream));
	clock_gettime(CLOCK_MONOTONIC_RAW, &odd_finish);
	printf("%s/%lu: %ldms\n", argv[3], i, ns2ms(timediff(even_finish, odd_finish)));
	fprintf(stderr, "Done. Shutting down...\n");

	cudaFree(pinned_hostmem);
	cudaFree(devmem_d2);
	cudaFree(devmem);
	return 0;
}

