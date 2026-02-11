/* Copyright 2021-2023 Joshua Bakita
 * Header for miscellaneous experimental helper functions.
 */

// cudaError_t and CUResult can both safely be cast to an unsigned int
static __thread unsigned int __SAFE_err;

// The very strange cast in these macros is to satisfy two goals at tension:
// 1. This file should be able to be included in non-CUDA-using files, and thus
//    should use no CUDA types outside of this macro.
// 2. We want to typecheck uses of these macros. The driver and runtime APIs
//    do not have identical error numbers and/or meanings, so runtime library
//    calls should use SAFE, and driver library calls should use SAFE_D.
// Our design allows typechecking while keeping a non-CUDA per-thread error var.

// For CUDA Runtime Library functions; typically those prefixed with `cuda`
#define SAFE(x) \
	if ((*(cudaError_t*)(&__SAFE_err) = (x)) != 0) { \
		printf("(%s:%d) CUDA error %d: %s i.e. \"%s\" returned by %s. Aborting...\n", \
		       __FILE__, __LINE__, __SAFE_err, cudaGetErrorName((cudaError_t)__SAFE_err), cudaGetErrorString((cudaError_t)__SAFE_err), #x); \
		exit(1); \
	}

// For CUDA Driver Library functions; typically those prefixed with just `cu`
#define SAFE_D(x) \
	if ((*(CUresult*)&(__SAFE_err) = (x)) != 0) { \
		const char* name; \
		const char* desc; \
		cuGetErrorName((CUresult)__SAFE_err, &name); \
		cuGetErrorString((CUresult)__SAFE_err, &desc); \
		printf("(%s:%d) CUDA error %d: %s i.e. \"%s\" returned by %s. Aborting...\n", \
		       __FILE__, __LINE__, __SAFE_err, name, desc, #x); \
		exit(1); \
	}

#define s2ns(s) ((s)*1000l*1000l*1000l)
#define ns2ms(s) ((s)/(1000l*1000l))

// Return the difference between two timestamps in nanoseconds
#define timediff(start, end) ((s2ns((end).tv_sec) + (end).tv_nsec) - (s2ns((start).tv_sec) + (start).tv_nsec))
#define time2ns(time) (s2ns((time).tv_sec) + (time).tv_nsec)
