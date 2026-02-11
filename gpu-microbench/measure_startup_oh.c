/* Copyright 2025 Joshua Bakita
 * Simple kernel that uses measure_launch_oh to measure how long CUDA
 * initialization takes.
 * Prints status to stderr and nothing to stdout.
 */
#include <time.h>   // clock_gettime()
#include <stdio.h>  // fprintf(), perror()
#include <stdlib.h> // putenv()
#include <string.h> // strcmp()
#include <unistd.h> // execve()
extern char **environ;

#include "testbench.h"

static char* child_argv_default[3] = {"./measure_launch_oh", "1", NULL};

int main(int argc, char **argv) {
	struct timespec start;

	if (argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h"))) {
		fprintf(stderr, "Usage: %s [-e environment var to apply to child (optional) | prog prog_arg...]\n", argv[0]);
		return 1;
	}

	char **child_argv = child_argv_default;
	if (argv[1]) {
		// Change env
		// (Use this to apply LD_PRELOAD to the child only)
		if (!strcmp(argv[1], "-e"))
			putenv(argv[2]);
		// Special exec (e.g., ./nvtaskset --gpc-list 0 ./measure_launch_oh 1)
		else
			child_argv = &argv[1];
	}

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	fprintf(stderr, "(%d) Starting '%s' at CLOCK_MONOTONIC_RAW = %ld ns\n",
			getpid(), child_argv[0], time2ns(start));
	if (execve(child_argv[0], child_argv, environ)) {
		perror("Unable to start subprogram");
		return 1;
	}

	return 0;
}
