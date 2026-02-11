.PHONY: all clean
NVCC ?= /usr/local/cuda/bin/nvcc

all: copy_contender mon_cross_ctx_copies

copy_contender: copy_contender.cu copy_testbench.h ../testbench.h
	$(NVCC) copy_contender.cu -o $@ -lcuda -g

mon_cross_ctx_copies: mon_cross_ctx_copies.cu copy_testbench.h ../testbench.h
	$(NVCC) mon_cross_ctx_copies.cu -o $@ -lcuda -lpthread -g

gl_copies: gl_copies.c
	$(CC) gl_copies.c -o $@ -lGL -lglut -g

clean:
	rm -f copy_contender mon_cross_ctx_copies gl_copies
