/* Copyright 2021-2023 Joshua Bakita
 * Header-only library that provides an ultra-low-overhead GPU timing function,
 * a copy monitoring kernel, and other useful sundries.
 */
#include <stdio.h>
#include <stdint.h>

#include "../testbench.h"

// 4k pages (4096 bytes)
#define PG_SZ 0x1000

// Compiles to 5 inline SASS instructions on sm_52
static __device__ inline uint64_t gclock64_fast(void) {
	uint32_t lo_bits;
	uint32_t hi_bits;
	uint32_t hi_bits_2;
	uint64_t ret;
	// Upper bits may rollover between our first and 2nd read
	asm volatile("mov.u32 %0, %%globaltimer_hi;" : "=r"(hi_bits));
	asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(lo_bits));
	asm volatile("mov.u32 %0, %%globaltimer_hi;" : "=r"(hi_bits_2));
	// If upper bits rolled over, lo_bits = 0
	lo_bits = (hi_bits != hi_bits_2) ? 0 : lo_bits;
	// As sm_52 SASS is naively 32-bit, the following ops get optimized out
	ret = hi_bits_2;
	ret <<= 32;
	ret |= lo_bits;
	return ret;
}

/* Old copy function. Copyright Nathan Otterness
// Returns the value of CUDA's global nanosecond timer.
// Compiles to 9 inline SASS instructions on sm_52
static __device__ inline uint64_t gclock64(void) {
  // Due to a bug in CUDA's 64-bit globaltimer, the lower 32 bits can wrap
  // around after the upper bits have already been read. Work around this by
  // reading the high bits a second time. Use the second value to detect a
  // rollover, and set the lower bits of the 64-bit "timer reading" to 0, which
  // would be valid, it's passed over during the duration of the reading. If no
  // rollover occurred, just return the initial reading.
  volatile uint64_t first_reading;
  volatile uint32_t second_reading;
  uint32_t high_bits_first;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(first_reading));
  high_bits_first = first_reading >> 32;
  asm volatile("mov.u32 %0, %%globaltimer_hi;" : "=r"(second_reading));
  if (high_bits_first == second_reading) {
    return first_reading;
  }
  // Return the value with the updated high bits, but the low bits set to 0.
  return ((uint64_t) second_reading) << 32;
}
*/

// This works by monitoring and recording when the last byte of each
// 4k page of the `devmem_to_watch` becomes non-zero.
// @param devmem_len Length of devmem_to_watch in bytes
// @param times      Arr to store fin time of each pg cpy. Len == devmem_len / PG_SZ
__global__ void watch_devmem(char* devmem_to_watch, int devmem_len, uint64_t* times, int* barrier) {
	volatile char* cursor = devmem_to_watch;
	// If more than one thread is launched, distribute work by offsetting threads
	if (blockDim.x > 1) {
		times += threadIdx.x;
		cursor += threadIdx.x * PG_SZ;
	}
	barrier[threadIdx.x] = 1;
	// Assume they show up in the right order
	while (cursor < devmem_to_watch + devmem_len) {
		// Wait until last byte of the page is written to
		while (!*(cursor + PG_SZ - 1))
			continue;
		*times = gclock64_fast(); // Log time
		times += blockDim.x;
		cursor += PG_SZ * blockDim.x; // Advance a page
	}
}

static inline char* human_readable_bytes(size_t bytes) {
	static char out[100];
	if (bytes > 1024.*1024*1024)
		snprintf(out, 100, "%.2fGiB", bytes/(1024.*1024*1024));
	else if (bytes > 1024.*1024)
		snprintf(out, 100, "%.2fMiB", bytes/(1024.*1024));
	else if (bytes > 1024.)
		snprintf(out, 100, "%.2fKiB", bytes/1024.);
	return out;
}
