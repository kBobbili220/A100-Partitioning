.PHONY: all copy_experiments clean
NVCC ?= /usr/local/cuda/bin/nvcc

all: constant_cycles_kernel preemption_logger copy_experiments \
     measure_launch_oh measure_startup_oh

constant_cycles_kernel: constant_cycles_kernel.cu testbench.h
	$(NVCC) constant_cycles_kernel.cu -o constant_cycles_kernel

preemption_logger: task_host_utilities.cu preemption_logger.cu testbench.h
	$(NVCC) preemption_logger.cu -o preemption_logger -g

measure_launch_oh: measure_launch_oh.cu testbench.h
	$(NVCC) measure_launch_oh.cu -o measure_launch_oh

measure_startup_oh: measure_startup_oh.c testbench.h
	$(NVCC) measure_startup_oh.c -o measure_startup_oh

copy_experiments:
	$(MAKE) -C $@

clean:
	rm -f constant_cycles_kernel preemption_logger measure_launch_oh \
	      measure_startup_oh
	$(MAKE) -C copy_experiments clean
