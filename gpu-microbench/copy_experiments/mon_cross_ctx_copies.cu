/* Copyright 2024 Joshua Bakita
 * Starts two copies of configurable size in seperate contexts at the same time
 * (by default) and logs the times that pages are copied to the GPU. Supports
 * monitoring copy progress from either the GPU or the CPU. All output times are
 * converted to ns.
 * Some assumptions:
 * - GPU clock ticks at a constant rate
 * - GPU clock ticks even while GPU is idle
 * - Different contexts view the same underlying GPU clock
 * - GPU and CPU clocks tick at the same rate
 *
 * BUGS:
 * - GPU and CPU clocks *do not* tick at the same rate
 * - CPU clocks may be at different offsets and rates on different cores
 * TODO:
 * - Disable migrations while consistent timestamps are needed
 * - When using GPU monitoring, synchronize clocks in each thread, and
 *   do this serially
 * - Support configuring copy direction
 *
 * Note that only CPU-side monitoring is used in the RTAS'24 paper, so this
 * tool is still correct and known-bug-free for the purposes of artifact
 * evaluation.
 */
#include "copy_testbench.h"
#include "../task_host_utilities.cu"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <cuda.h>
#include <pthread.h>
#include <argp.h>

// Data for the help text(s)
const char* maintainer = "<jbakita@cs.unc.edu>";
const char* version = "mon_cross_ctx_copies 2023.06";
const char* desc =
    "Copy data to, from, or between GPU(s) and log the time that each block "
    "of the copy completes. Data need not be transferred sequentially for "
    "transfer times to be properly recorded.";

const struct argp_option opts[] = {
    {"copy_size", 's', "NUM_PAGES", 0, "Number of 4KiB pages to copy"},
    {"gpu", 'g', /*"DIRECTION"*/0, 0, "Add a GPU-monitored copy thread (can be duplicated)"},
    {"cpu", 'c', /*"DIRECTION"*/0, 0, "Add a CPU-monitored copy thread (can be duplicated)"},
    {0}
};

// CPU-only monitoring can only go a fraction of the speed of GPU monitoring.
// This sets how much slower it should go.
// WARNING: Setting this too low will result in low clock resolution, but
//          setting this too high will result in low unique sample count.
static const int CPU_MON_DIVISOR = 50;
// A single GPU thread cannot log page times as fast as they can be copied, so
// spread the work across multiple threads.
// WARNING: Setting this too high will result in extra memory contention, but
//          setting this too low will result in inaccurate copy logs.
// Known working settings:
// | Chip:  | Num threads:
// | GV100  | 8
// | GV11b  | 32
static const int GPU_MON_THREADS = 32;
// If delay start is enabled on any thread, how great is the delay?
static const struct timespec DELAY_TIME = {0, 50*1000l*1000l}; // 50ms (use 30ms for GV100)
// Should each GPU-copy-monitoring thread check for GPU clock skew?
static const bool SKEW_CHECK = false;

// Size of each copy to the GPU, in bytes (64-bit, as copies may be > 4GiB)
static uint64_t COPY_SIZE;
// GPU time offset. CPU time = GPU time - offset
static uint64_t GPU_TIME_OFFSET;
// Copy start barrier
static unsigned int READY = 0;

typedef enum {
	CPU_MONITORING,
	GPU_MONITORING,
	DELAY_START,  // For DELAY_TIME as set above
} my_config_t;

typedef struct {
	int is_ready;
	my_config_t config;
} copy_thread_args_t;

// Repeatedly update a time counter in the last 8 bytes of every Nth page, where N is CPU_MON_DIVISOR
void cpu_copy_mon(int loops, char* cpu_mem) {
	struct timespec curr_time;
	uint64_t curr_time_64;

	for (int i = 0; i < loops; i++) {
		clock_gettime(CLOCK_MONOTONIC_RAW, &curr_time);
		curr_time_64 = s2ns(curr_time.tv_sec);
		curr_time_64 += curr_time.tv_nsec;
		// Updating pages is slower than the CE, so sparely timestamp
		for (int offset = 0; offset < COPY_SIZE; offset += PG_SZ * CPU_MON_DIVISOR - 8)
			*(uint64_t*)(cpu_mem + offset) = curr_time_64;
	}
}

// Run a single copy in it's own context (intended to be run as a thread)
void* copy_thread(void* args_raw) {
	CUcontext ctx;
	int dev, i;
	int *barrier;
	int *barrier_dev_ptr; // On Pascal+ this will be the same as *barrier
	char *pinned_hostmem, *devmem;
	uint64_t *devtimes, *times;
	cudaStream_t stream1, stream2;
	copy_thread_args_t *args = (copy_thread_args_t*)args_raw;

	bool DELAY = args->config == DELAY_START;
	bool GPU_COMPUTE = args->config == GPU_MONITORING;

	// Explictly create a context (avoids creating a primary context implictly)
	// This has been verified on CUDA 11.1 to give each thread a different context
	// handle
	SAFE(cudaGetDevice(&dev));
	SAFE_D(cuCtxCreate(&ctx, 0, 0, dev));

	uint64_t dev_ns, dev_ns2;
	double host_s, host_s2;
	if (GPU_COMPUTE && SKEW_CHECK)
		InternalReadGPUNanoseconds(dev, &host_s, &dev_ns);

	SAFE(cudaStreamCreate(&stream1));
	SAFE(cudaStreamCreate(&stream2));

	if (GPU_COMPUTE) {
		SAFE(cudaHostAlloc(&barrier, sizeof(int) * GPU_MON_THREADS, cudaHostAllocMapped));
		SAFE(cudaMemset(barrier, 0, sizeof(int) * GPU_MON_THREADS));
		SAFE(cudaHostGetDevicePointer(&barrier_dev_ptr, barrier, 0));
	}

	SAFE(cudaHostAlloc(&pinned_hostmem, COPY_SIZE, cudaHostAllocDefault));
	for (i = 0; i < COPY_SIZE; i++)
		// Don't allow 0 so that the copy detection logic works
		pinned_hostmem[i] = max((rand() & 0xff), 1);
	SAFE(cudaMalloc(&devmem, COPY_SIZE));
	SAFE(cudaMemset(devmem, 0, COPY_SIZE));

	SAFE(cudaMalloc(&devtimes, 8 * COPY_SIZE / PG_SZ));
	times = (uint64_t*)malloc(8 * COPY_SIZE / PG_SZ);
	if (!times) {
		fprintf(stderr, "Out of memory! Exiting...\n");
		exit(1);
	}

	if (GPU_COMPUTE) {
		// 1 thread can't go quick enough, even on the GV100
		watch_devmem<<<1,GPU_MON_THREADS,0,stream2>>>(devmem, COPY_SIZE, devtimes, barrier_dev_ptr);
		// Wait for monitor(s) to initialize
		bool ready = false;
		while (!ready) {
			ready = true;
			for (int i = 0; i < GPU_MON_THREADS; i++)
				ready &= barrier[i];
		}
	} else {
		if ((COPY_SIZE / PG_SZ) % CPU_MON_DIVISOR != 0) {
			fprintf(stderr, "copy_size must be divisible by %d when using CPU-monitored copy threads.\n", CPU_MON_DIVISOR);
			exit(1);
		}
	}

	// Tell our parent we're ready
	args->is_ready = 1;
	// Wait for our parent to tell us to go (spinning here should also cause
	// Linux's load-balancing logic to implictly move each monitoring thread to
	// a separate core)
	while (!READY)
		continue;

	if (DELAY)
		nanosleep(&DELAY_TIME, NULL);

	if (!GPU_COMPUTE) {
		// If CPU-only monitoring, populate initial times
		cpu_copy_mon(1, pinned_hostmem);
	}

	SAFE(cudaMemcpyAsync(devmem, pinned_hostmem, COPY_SIZE, cudaMemcpyHostToDevice, stream1));
	// XXX start
	//if (GPU_COMPUTE) {
	//	const struct timespec DELAY_TIME = {0, 50*1000l*1000l}; // 50ms
	//	nanosleep(&DELAY_TIME, NULL);
	//	watch_devmem<<<1,8,0,stream2>>>(devmem, COPY_SIZE, devtimes, barrier);
	//}
	// XXX end

	if (GPU_COMPUTE) {
		// Wait for copy monitor (and hence copy) to complete
		SAFE(cudaStreamSynchronize(stream2));
	} else {
		// Guess number of needed CPU-only monitoring cycles (this heuristic
		// ensures that monitoring runs for roughly the same amount of time,
		// no matter the timestamping granularity).
		cpu_copy_mon(CPU_MON_DIVISOR * COPY_SIZE / PG_SZ, pinned_hostmem);
		// Make sure that the copy finished in case we guessed small
		SAFE(cudaStreamSynchronize(stream1));
	}

	// Pull back times from the GPU
	if (GPU_COMPUTE) {
		SAFE(cudaMemcpy(times, devtimes, 8 * COPY_SIZE / PG_SZ, cudaMemcpyDeviceToHost));
		// Convert GPU times to CPU times
		for (int i = 0; i < COPY_SIZE / PG_SZ; i++)
			times[i] -= GPU_TIME_OFFSET;
	} else {
		SAFE(cudaMemcpy(pinned_hostmem, devmem, COPY_SIZE, cudaMemcpyDeviceToHost));
		int times_idx = 0;
		// CPU-only monitoring has a fraction the # samples, so broadcast
		for (int offset = PG_SZ * CPU_MON_DIVISOR - 8; offset < COPY_SIZE; offset += PG_SZ * CPU_MON_DIVISOR - 8) {
			uint64_t time = *(uint64_t*)(pinned_hostmem + offset);
			for (int i = 0; i < CPU_MON_DIVISOR; i++) {
				times[times_idx++] = time;
			}
		}
		// With CPU-only monitoring, we also record the very first page seperately
		times[0] = *(uint64_t*)pinned_hostmem;
	}

	if (GPU_COMPUTE && SKEW_CHECK) {
		InternalReadGPUNanoseconds(dev, &host_s2, &dev_ns2);
		double host_diff = s2ns(host_s2 - host_s);
		double dev_diff = dev_ns2 - dev_ns;
		fprintf(stderr, "GPU clock ran at rate %f of CPU clock (%f ticks GPU vs %f ticks CPU)\n", dev_diff/host_diff, dev_diff, host_diff);
	}

	SAFE(cudaFreeHost(pinned_hostmem));
	SAFE(cudaFree(devtimes));
	SAFE(cudaFree(devmem));

	return times;
}

const int MAX_THREADS = 32;

typedef struct {
	int num_threads;
	copy_thread_args_t thread_args[MAX_THREADS];
} global_args_t;

static error_t arg_parser(int key, char* arg, struct argp_state *state) {
	global_args_t* g_args = (global_args_t*)state->input;
	switch (key) {
		case 's':
			if (atol(arg) < CPU_MON_DIVISOR * 2)
				argp_error(state, "Please specify a larger copy size. It must be at least %d pages for accurate tracking.\n", CPU_MON_DIVISOR * 2);
			COPY_SIZE = strtoull(arg, NULL, 0) * PG_SZ;
			break;
		case 'g':
			if (g_args->num_threads == MAX_THREADS)
				argp_error(state, "Please specify less than %d monitoring threads, or increase the compiled in MAX_THREADS constraint.\n", MAX_THREADS);
			g_args->thread_args[g_args->num_threads++] = {0, GPU_MONITORING};
			break;
		case 'c':
			if (g_args->num_threads == MAX_THREADS)
				argp_error(state, "Please specify less than %d monitoring threads, or increase the compiled in MAX_THREADS constraint.\n", MAX_THREADS);
			g_args->thread_args[g_args->num_threads++] = {0, CPU_MONITORING};
			break;
		default:
			return ARGP_ERR_UNKNOWN;
	}
	return 0;
}

int main(int argc, char**argv) {
	int tmp, dev = 0;
	CUdevice dev_itrl;
	uint64_t *ctx_times[MAX_THREADS] = {0};
	pthread_t t[MAX_THREADS];
	global_args_t g_args = {0};

	struct argp argp = {opts, arg_parser, 0, desc};
	argp_parse(&argp, argc, argv, 0, 0, &g_args);

	if (g_args.num_threads == 0) {
		fprintf(stderr, "At least one copy thread must be specified with --gpu or --cpu arguments.\n");
		return 3;
	}

	fprintf(stderr, "(%d) Synchronizing clocks and initializing copy threads...\n", getpid());

	// Temporarially initialize CUDA to query device attributes
	SAFE_D(cuInit(0));
	SAFE_D(cuDeviceGet(&dev_itrl, dev));
	// Due to some laziness in how we handle barriers, this flag needs to be true
	/// XXX: Still seems to work fine if it isn't???
	SAFE_D(cuDeviceGetAttribute(&tmp, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, dev_itrl));
	if (!tmp) {
		fprintf(stderr, "Unsupported GPU. It must be possible to map host (CPU)"
				" DRAM into the GPU virtual address space for accurate clock "
				"synchronization. Exiting...\n");
		return 1;
	}
	// Terminate the context used for attrib check so it's not accidentially
	// reused in subprocesses
	SAFE_D(cuDevicePrimaryCtxRelease(dev_itrl));

	double d2h_scale, host_s;
	uint64_t dev_ns;
	// Get the core-specific offset of GPU time from CPU time, and the
	// core-specific difference in tick rates (typical variance of -13 to 60
	// microseconds per second).
	// XXX: This is not sufficient for time synchronization, see "BUGS" at the
	//      top of this file.
	// XXX: This creates an implict context, but should reuse the above.
	d2h_scale = InternalGetGPUTimerScale(dev);
	InternalReadGPUNanoseconds(dev, &host_s, &dev_ns);
	if (d2h_scale == -1 || (host_s == 0 && !dev_ns)) {
		fprintf(stderr, "Unabled to synchronize time with the GPU. Aborting...\n");
		return 1;
	}
	GPU_TIME_OFFSET = dev_ns - s2ns(host_s);
	// Necessary to synchronize experiments running on different CPU cores
	// (as CPU clocks are only semi-synchronized)
	fprintf(stderr, "(%d) CPU clock - GPU clock: %ld tick gap\n", getpid(), (long)s2ns(host_s) - dev_ns);
	fprintf(stderr, "(%d) 1 CPU tick/1 GPU tick: %.9f\n", getpid(), d2h_scale);

	// Copy buffers are filled with random numbers. Seed the RNG.
	srand(0);
	// Spawn and wait for children to intialize
	for (int tid = 0; tid < g_args.num_threads; tid++) {
		pthread_create(&t[tid], NULL, copy_thread, &g_args.thread_args[tid]);
		while (!g_args.thread_args[tid].is_ready)
			continue;
	}

	fprintf(stderr, "(%d) Initialization completed. Press enter to start copies...", getpid());
	getc(stdin); // Wait for user
	// Tell children initialization is done and that they can go
	READY = 1;

	// Wait for threads to finish, and determine the earliest recorded time
	uint64_t smallest = UINT64_MAX;
	for (int tid = 0; tid < g_args.num_threads; tid++) {
		pthread_join(t[tid], (void**)&ctx_times[tid]);
		smallest = min(smallest, ctx_times[tid][0]);
	}
	// Rebase times to 0 and print
	for (int tid = 0; tid < g_args.num_threads; tid++) {
		// Summary to stderr
		fprintf(stderr, "%lu, %lu\n", ctx_times[tid][0] - smallest, ctx_times[tid][COPY_SIZE / PG_SZ - 1] - smallest);
		// Full dataset to stdout
		for (tmp = 0; tmp < COPY_SIZE / PG_SZ - 1; tmp++)
			printf("%lu, ", ctx_times[tid][tmp]);
		printf("%lu\n", ctx_times[tid][COPY_SIZE / PG_SZ - 1]);
	}
	return 0;
}
