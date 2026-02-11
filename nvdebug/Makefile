obj-m += nvdebug.o
nvdebug-objs = runlist_procfs.o device_info_procfs.o runlist.o mmu.o \
               nvdebug_entry.o bus.o nvdebug_linux.o copy_topology_procfs.o
KBUILD_CFLAGS += -DGIT_HASH=\"$(shell git --git-dir=$(PWD)/.git rev-parse --short HEAD)\" \
		 -Wno-missing-prototypes -Wno-missing-declarations
# -mfentry above if not building due to mcount missing

all:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean

nvdebug_user.so: runlist.c mmu.c bus.c nvdebug_user.c
	gcc $< -shared -o $@ $(KBULID_CFLAGS)
