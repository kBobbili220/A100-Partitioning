/* Copyright 2024 Joshua Bakita
 * SPDX-License-Identifier: MIT
 */

#include <linux/device.h>  // For struct device, bus_find_device*(), struct bus_type
#include <linux/interrupt.h> // For hooking the nvidia driver interrupts
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/pci.h>  // For PCI device scanning
#include <linux/platform_device.h>  // For platform_device struct
#include <linux/proc_fs.h>  // So we can set up entries in /proc

#include "nvdebug_linux.h"
#include "stubs.h"

// Enable to intercept and log GPU interrupts. Historically used to benchmark
// interrupt latency.
#define INTERRUPT_DEBUG

// MIT is GPL-compatible. We need to be GPL-compatible for symbols like
// platform_bus_type or bus_find_device_by_name...
MODULE_LICENSE("Dual MIT/GPL");
MODULE_AUTHOR("Joshua Bakita");
MODULE_DESCRIPTION("A scheduling debugging module for NVIDIA GPUs");

// runlist_procfs.c
extern struct file_operations runlist_file_ops;
extern struct file_operations preempt_tsg_file_ops;
extern struct file_operations disable_channel_file_ops;
extern struct file_operations enable_channel_file_ops;
extern struct file_operations wfi_preempt_channel_file_ops;
extern struct file_operations cta_preempt_channel_file_ops;
extern struct file_operations cil_preempt_channel_file_ops;
extern struct file_operations resubmit_runlist_file_ops;
extern struct file_operations preempt_runlist_file_ops;
extern struct file_operations ack_bad_tsg_file_ops;
extern struct file_operations map_mem_chid_file_ops;
extern struct file_operations map_mem_ctxid_file_ops;
extern struct file_operations switch_to_tsg_file_ops;
// device_info_procfs.c
extern struct file_operations device_info_file_ops;
extern struct file_operations nvdebug_read_reg32_file_ops;
extern struct file_operations nvdebug_read_reg_range_file_ops;
extern struct file_operations nvdebug_read_part_file_ops;
extern struct file_operations local_memory_file_ops;
// copy_topology_procfs.c
extern struct file_operations copy_topology_file_ops;

struct nvdebug_state g_nvdebug_state[NVDEBUG_MAX_DEVICES];
unsigned int g_nvdebug_devices = 0;

// Starting in Kernel 5.6, proc_ops is required instead of file_operations.
// As file_operations is larger than proc_ops, we can overwrite the memory
// backing the file_operations struct to follow the proc_ops layout, and then
// cast on newer kernels.
// We use the last byte of the file_operations struct to flag that the memory
// layout has been rearranged.
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5,6,0)
const struct proc_ops* compat_ops(const struct file_operations* ops) {
	struct proc_ops new_ops = {};
	// Don't re-layout if it's already been done
	if (*((uint8_t*)(ops + 1) - 1))
		return (struct proc_ops*)ops;
	new_ops.proc_open = ops->open;
	new_ops.proc_read = ops->read;
	new_ops.proc_write = ops->write;
	new_ops.proc_lseek = ops->llseek;
	new_ops.proc_release = ops->release;
	memcpy((void*)ops, &new_ops, sizeof(new_ops));
	// Flag re-layout as complete in last byte of structure
	*((uint8_t*)(ops + 1) - 1) = 1;
	return (struct proc_ops*)ops;
}
#else
const struct file_operations* compat_ops(const struct file_operations* ops) {
	return ops;
}
#endif

#ifdef INTERRUPT_DEBUG

void nvdebug_fifo_intr(struct nvdebug_state *g) {
	uint32_t fifo_intr_mask;// = nvdebug_readl(g, 0x02100); // PFIFO_INTR_0
	fifo_intr_mask = nvdebug_readl(g, 0x02100); // PFIFO_INTR_0
	if (fifo_intr_mask & 1 << 0)
		printk(KERN_INFO "[nvdebug]   - Interrupt BIND_ERROR.\n");
	if (fifo_intr_mask & 1 << 1)
		printk(KERN_INFO "[nvdebug]   - Interrupt CTXSW_TIMEOUT.\n");
	if (fifo_intr_mask & 1 << 4)
		printk(KERN_INFO "[nvdebug]   - Interrupt RUNLIST_IDLE.\n");
	if (fifo_intr_mask & 1 << 5)
		printk(KERN_INFO "[nvdebug]   - Interrupt RUNLIST_AND_ENG_IDLE.\n");
	if (fifo_intr_mask & 1 << 6)
		printk(KERN_INFO "[nvdebug]   - Interrupt RUNLIST_ACQUIRE.\n");
	if (fifo_intr_mask & 1 << 7)
		printk(KERN_INFO "[nvdebug]   - Interrupt RUNLIST_ACQUIRE_AND_ENG_IDLE.\n");
	if (fifo_intr_mask & 1 << 8)
		printk(KERN_INFO "[nvdebug]   - Interrupt SCHED_ERROR.\n");
	if (fifo_intr_mask & 1 << 16)
		printk(KERN_INFO "[nvdebug]   - Interrupt CHSW_ERROR.\n");
	if (fifo_intr_mask & 1 << 23)
		printk(KERN_INFO "[nvdebug]   - Interrupt MEMOP_TIMEOUT.\n");
	if (fifo_intr_mask & 1 << 24)
		printk(KERN_INFO "[nvdebug]   - Interrupt LB_ERROR.\n");
	if (fifo_intr_mask & 1 << 25) // OLD; Pascal
		printk(KERN_INFO "[nvdebug]   - Interrupt REPLAYABLE_FAULT_ERROR.\n");
	if (fifo_intr_mask & 1 << 27) // OLD; Pascal
		printk(KERN_INFO "[nvdebug]   - Interrupt DROPPED_MMU_FAULT.\n");
	if (fifo_intr_mask & 1 << 28) { // On Pascal, this is MMU_FAULT
		if (g->chip_id <= NV_CHIP_ID_VOLTA) // MMU_FAULT on Pascal (nvgpu, l4t/l4t-r28.1:drivers/gpu/nvgpu/include/nvgpu/hw/gp10b/hw_fifo_gp10b.h)
			printk(KERN_INFO "[nvdebug]   - Interrupt MMU_FAULT.\n");
		else // Repurposed starting with Turing: open-gpu-doc/manuals/turing/tu104/dev_fifo.ref.txt
			printk(KERN_INFO "[nvdebug]   - Interrupt TSG_PREEMPT_COMPLETE.\n");
	}
	if (fifo_intr_mask & 1 << 29)
		printk(KERN_INFO "[nvdebug]   - Interrupt PBDMA_INTR.\n");
	if (fifo_intr_mask & 1 << 30) {
		printk(KERN_INFO "[nvdebug]   - Interrupt RUNLIST_EVENT.\n");
		uint32_t fifo_runlist_intr_mask = nvdebug_readl(g, 0x02A00); // PFIFO_INTR_RUNLIST
		printk(KERN_INFO "[nvdebug]     - Event %#x.\n", fifo_runlist_intr_mask);
	}
	if (fifo_intr_mask & 1 << 31)
		printk(KERN_INFO "[nvdebug]   - Interrupt CHANNEL_INTR.\n");
}

irqreturn_t nvdebug_irq_tap(int irq_num, void * dev) {
	struct nvdebug_state *g = dev;
	u64 time = ktime_get_raw_ns(); // CLOCK_MONOTONTIC_RAW
	// NV_PMC_INTR does not exist on Ada, so use NV_FUNC_PRIV_CPU_INTR_TOP
	// Note that this also appears to exist on Turing
	if (g->chip_id >= NV_CHIP_ID_TURING) {//AMPERE) {
		int i;
		// Despite being an indexed register, it is only documented to have on, and could only support two
		uint32_t intr_mask0 = nvdebug_readl(g, NV_VIRTUAL_FUNCTION_FULL_PHYS_OFFSET + 0x1600); // NV_FUNC_PRIV_CPU_INTR_TOP(0)
		uint32_t intr_mask1 = nvdebug_readl(g, NV_VIRTUAL_FUNCTION_FULL_PHYS_OFFSET + 0x1604); // NV_FUNC_PRIV_CPU_INTR_TOP(1)
		printk(KERN_INFO "[nvdebug] Interrupt on IRQ %d with CPU_INTR_TOP(0) %#010x, ...(1) %#010x @ %llu.\n", irq_num, intr_mask0, intr_mask1, time);
		for (i = 0; i < 8; i++) {
			uint32_t leaf = nvdebug_readl(g, NV_VIRTUAL_FUNCTION_FULL_PHYS_OFFSET + 0x1000 + i*4); // NV_FUNC_PRIV_CPU_INTR_LEAF(0) to ...(7)
			if (leaf)
				printk(KERN_INFO "[nvdebug] - Interrupt leaf %d: %#010x\n", i, leaf);
			// 131-133 & 64 are faults on tu104??? (open-gpu-doc/manuals/turing/tu104/pri_mmu_hub.ref.txt)
			if (136 / 32 == i && 1 << (136 % 32) & leaf) // PFIFO0 ga100
				printk(KERN_INFO "[nvdebug] - Interrupt on PFIFO0.\n");
			if (137 / 32 == i && 1 << (137 % 32) & leaf) // PFIFO1 ga100
				printk(KERN_INFO "[nvdebug] - Interrupt on PFIFO1.\n");
			if (148 / 32 == i && 1 << (148 % 32) & leaf) // TIMER ga100
				printk(KERN_INFO "[nvdebug] - Interrupt on PTIMER.\n");
			if (152 / 32 == i && 1 << (152 % 32) & leaf) // PMU ga100
				printk(KERN_INFO "[nvdebug] - Interrupt on PMU.\n");
			if (156 / 32 == i && 1 << (156 % 32) & leaf) { // PBUS ga100
				printk(KERN_INFO "[nvdebug] - Interrupt on PBUS.\n");
				uint32_t bus_intr = nvdebug_readl(g, 0x1100); // BUS_INTR_0
				if (bus_intr & 1 << 2) {
					// use timer_pri_timeout_save_0_r
					uint32_t SAVE_0 = nvdebug_readl(g, 0x00009084); // NV_PTIMER_PRI_TIMEOUT_SAVE_0
					printk(KERN_INFO "[nvdebug]   - Interrupt PRI_FECSERR on %s to address %#010x %stargeting FECS.\n", SAVE_0 & 0x2 ? "write" : "read", SAVE_0 & 0x00fffffc, SAVE_0 & 0x80000000 ? "" : "not ");
					uint32_t SAVE_1 = nvdebug_readl(g, 0x00009088); // NV_PTIMER_PRI_TIMEOUT_SAVE_1
					if (SAVE_1)
						printk(KERN_INFO "[nvdebug]     Data written: %#010x\n", SAVE_1);
					uint32_t errcode = readl(g->regs + 0x0000908C); // NV_PTIMER_PRI_TIMEOUT_FECS_ERRCODE
					if (errcode)
						printk(KERN_INFO "[nvdebug]     FECS Error Code: %#010x\n", errcode);
					// badf5040 is a "client error" (0) of "no such address" (40)
					// See linux-nvgpu/drivers/gpu/nvgpu/hal/priv_ring/priv_ring_ga10b_fusa.c
					// for how to decode.
				}
				if (bus_intr & 1 << 3)
					printk(KERN_INFO "[nvdebug]   - Interrupt PRI_TIMEOUT.\n");
				if (bus_intr & 1 << 4)
					printk(KERN_INFO "[nvdebug]   - Interrupt FB_REQ_TIMEOUT.\n");
				if (bus_intr & 1 << 5)
					printk(KERN_INFO "[nvdebug]   - Interrupt FB_ACK_TIMEOUT.\n");
				if (bus_intr & 1 << 6)
					printk(KERN_INFO "[nvdebug]   - Interrupt FB_ACK_EXTRA.\n");
				if (bus_intr & 1 << 7)
					printk(KERN_INFO "[nvdebug]   - Interrupt FB_RDATA_TIMEOUT.\n");
				if (bus_intr & 1 << 8)
					printk(KERN_INFO "[nvdebug]   - Interrupt FB_RDATA_EXTRA.\n");
				if (bus_intr & 1 << 26)
					printk(KERN_INFO "[nvdebug]   - Interrupt SW.\n");
				if (bus_intr & 1 << 27)
					printk(KERN_INFO "[nvdebug]   - Interrupt POSTED_DEADLOCK_TIMEOUT.\n");
				if (bus_intr & 1 << 28)
					printk(KERN_INFO "[nvdebug]   - Interrupt MPMU.\n");
				if (bus_intr & 1 << 31)
					printk(KERN_INFO "[nvdebug]   - Interrupt ACCESS_TIMEOUT.\n");
			}
			if (158 / 32 == i && 1 << (158 % 32) & leaf) // PRIV_RING ga100
				printk(KERN_INFO "[nvdebug] - Interrupt on PRIV_RING.\n");
			if (192 / 32 == i && 1 << (192 % 32) & leaf) // LEGACY_ENGINE_STALL ga100
				printk(KERN_INFO "[nvdebug] - Interrupt on LEGACY_ENGINE_STALL.\n");
			if (160 / 32 == i && 1 << (160 % 32) & leaf) { // (likely) rl0 ga100
				printk(KERN_INFO "[nvdebug] - Interrupt on RUNLIST0.\n");
				uint32_t off;
				get_runlist_ram(g, 0, &off);
				uint32_t rl_intr = nvdebug_readl(g, off+0x100); 
				printk(KERN_INFO "[nvdebug]   - RUNLIST_INTR_0: %#x\n", rl_intr);
				if (1 << 12 && rl_intr) { // BAD_TSG
					printk(KERN_INFO "[nvdebug]    - BAD_TSG: %#x\n", nvdebug_readl(g, off+0x174));
				}
			}
			// Also getting 160, 161, and 162

			//uint32_t off;
			//get_runlist_ram(g, 12, &off);
			//printk(KERN_INFO "[nvdebug] - rl10 vector id 0 is %x\n", nvdebug_readl(g, off+0x160)); // NV_RUNLIST_INTR_VECTORID(0)
			// 160 is rl0 (C/G, LCE0, LCE1) vector id 0
			// 168 is rl11 (LCE3) vector id 0
			// 169 is rl12 (LCE4) vector id 0
			// 171 is rl1 (SEC) vector id 0
			// 176 is rl10 (LCE2) vector id 0
			// 224 is rl0 vector id 1
			// Only some interrupt vectors are hardcoded
		}
		// each subtree has two leafs? Each bit at the top corresponds to a subtree?
		// So, if bit 0 is set, that means subtree 0 (concept) and leaves 0 and 1
		// So, if bit 1 is set, that means subtree 1 (concept) and leaves 2 and 3
		// the #define'd interrupt vectors all seem to fall in the lower leaf of subtree 2,
		// except for INTR_HUB_ACCESS_CNTR_INTR_VECTOR is in the lower leaf of subtree 1
		if (g->chip_id >= NV_CHIP_ID_AMPERE)
			return IRQ_NONE;
	}
	uint32_t intr_mask = nvdebug_readl(g, 0x0100); // NV_PMC_INTR
	printk(KERN_INFO "[nvdebug] Interrupt on IRQ %d with MC_INTR %#010x @ %llu.\n", irq_num, intr_mask, time);
	// IDs likely changed Ampere+
	//if (g->chip_id >= NV_CHIP_ID_AMPERE) {
		// CIC is central interrupt controller
		// the u32 passed around nvgpu cic functions is one of the 
		// enable is nvgpu_cic_mon_intr_stall_unit_config(unit)
		// - Calls intr_stall_unit_config(unit)
		//  - for ga, calls unit = ga10b_intr_map_mc_stall_unit_to_intr_unit(unit) (doesn't do much)
		//  - for ga, calls nvgpu_cic_mon_intr_get_unit_info()
		//   - Does *subtree = g->mc.intr_unit_info[unit].subtree;
		//          *subtree_mask = g->mc.intr_unit_info[unit].subtree_mask;
		//  - for ga, calls ga10b_intr_config() w/ subtree info
		//uint32_t intr_stats = nvdebug_readl(g, 1600
		//return IRQ_NONE;
	//}
	if (intr_mask & 1 << 5)
		printk(KERN_INFO "[nvdebug] - Interrupt on LCE0.\n");
	if (intr_mask & 1 << 6)
		printk(KERN_INFO "[nvdebug] - Interrupt on LCE1.\n");
	if (intr_mask & 1 << 7)
		printk(KERN_INFO "[nvdebug] - Interrupt on LCE2.\n");
	if (intr_mask & 1 << 8) {
		printk(KERN_INFO "[nvdebug] - Interrupt on PFIFO.\n");
		nvdebug_fifo_intr(g);
	}
	if (intr_mask & 1 << 9) {
		printk(KERN_INFO "[nvdebug] - Interrupt on HUB.\n"); // "replayable_fault_pending" in nvgpu on Pascal, "HUB" on Volta+
		// on tu104, if vector is one of the below set in new-style interrupt vector, then MMU fault
		// - info_fault (134)
		// - nonreplay_fault error (133)
		// - nonreplay_fault notify (132)
		// - replay_fault error (131)
		// - replay_fault notify (64)
		// (but the above fault vectors are configurable)
		// if it's ecc_error, then not mmu error
		// Default fault vectors from open-gpu-doc/manuals/turing/tu104/pri_mmu_hub.ref.txt
		// Turing through (at least) Ampere (per nvgpu)

		// on gv100, parse fb_niso_intr_r 0x00100a20U, where bits:
		// - hub_access_counter notify (0)
		// - hub_access_counter error (1)
		// - replay_fault notify (27)
		// - replay_fault overflow (28)
		// - nonreplay_fault notify (29)
		// - nonreplay_fault overflow (30)
		// - other_fault notify (31)
		// Volta through Turing (per nvgpu)

		// On Pascal, it looks like it's a property of fifo_intr_0???
		if (g->chip_id < NV_CHIP_ID_VOLTA)
			nvdebug_fifo_intr(g);
	}
	if (intr_mask & 1 << 10)
		printk(KERN_INFO "[nvdebug] - Interrupt on LCE3.\n");
	if (intr_mask & 1 << 11)
		printk(KERN_INFO "[nvdebug] - Interrupt on LCE4.\n");
	if (intr_mask & 1 << 12) {
		printk(KERN_INFO "[nvdebug] - Interrupt on Graphics/Compute.\n");
		// Kepler through (at least) Ampere
		// From open-gpu-doc/manuals/volta/gv100/dev_graphics.ref.txt
		uint32_t graph_intr_mask = nvdebug_readl(g, 0x400100); // NV_PGRAPH_INTR
		if (graph_intr_mask & 1 << 0)
			printk(KERN_INFO "[nvdebug]   - Interrupt NOTIFY.\n");
		if (graph_intr_mask & 1 << 1)
			printk(KERN_INFO "[nvdebug]   - Interrupt SEMAPHORE.\n");
		if (graph_intr_mask & 1 << 4)
			printk(KERN_INFO "[nvdebug]   - Interrupt ILLEGAL_METHOD.\n");
		if (graph_intr_mask & 1 << 5)
			printk(KERN_INFO "[nvdebug]   - Interrupt ILLEGAL_CLASS.\n");
		if (graph_intr_mask & 1 << 6)
			printk(KERN_INFO "[nvdebug]   - Interrupt ILLEGAL_NOTIFY.\n");
		if (graph_intr_mask & 1 << 7)
			printk(KERN_INFO "[nvdebug]   - Interrupt DEBUG_METHOD.\n");
		if (graph_intr_mask & 1 << 8)
			printk(KERN_INFO "[nvdebug]   - Interrupt FIRMWARE_METHOD.\n");
		if (graph_intr_mask & 1 << 16)
			printk(KERN_INFO "[nvdebug]   - Interrupt BUFFER_NOTIFY.\n");
		if (graph_intr_mask & 1 << 19)
			printk(KERN_INFO "[nvdebug]   - Interrupt FECS_ERROR.\n");
		if (graph_intr_mask & 1 << 20)
			printk(KERN_INFO "[nvdebug]   - Interrupt CLASS_ERROR.\n");
		if (graph_intr_mask & 1 << 21)
			printk(KERN_INFO "[nvdebug]   - Interrupt EXCEPTION.\n");
	}
	if (intr_mask & 1 << 13)
		printk(KERN_INFO "[nvdebug] - Interrupt on PFB.\n");
	if (intr_mask & 1 << 15)
		printk(KERN_INFO "[nvdebug] - Interrupt on SEC.\n");
	if (intr_mask & 1 << 16)
		printk(KERN_INFO "[nvdebug] - Interrupt on NVENC0.\n");
	if (intr_mask & 1 << 17)
		printk(KERN_INFO "[nvdebug] - Interrupt on NVDEC0.\n");
	if (intr_mask & 1 << 18)
		printk(KERN_INFO "[nvdebug] - Interrupt on THERMAL.\n");
	if (intr_mask & 1 << 19)
		printk(KERN_INFO "[nvdebug] - Interrupt on HDACODEC.\n");
	if (intr_mask & 1 << 20)
		printk(KERN_INFO "[nvdebug] - Interrupt on PTIMER.\n");
	if (intr_mask & 1 << 21)
		printk(KERN_INFO "[nvdebug] - Interrupt on PMGR.\n");
	if (intr_mask & 1 << 22)
		printk(KERN_INFO "[nvdebug] - Interrupt on IOCTRL.\n");
	if (intr_mask & 1 << 23)
		printk(KERN_INFO "[nvdebug] - Interrupt on DFD.\n");
	if (intr_mask & 1 << 24)
		printk(KERN_INFO "[nvdebug] - Interrupt on PMU.\n");
	if (intr_mask & 1 << 25)
		printk(KERN_INFO "[nvdebug] - Interrupt on LTC.\n");
	if (intr_mask & 1 << 26)
		printk(KERN_INFO "[nvdebug] - Interrupt on PDISP.\n");
	if (intr_mask & 1 << 27)
		printk(KERN_INFO "[nvdebug] - Interrupt on GSP.\n");
	if (intr_mask & 1 << 28)
		printk(KERN_INFO "[nvdebug] - Interrupt on PBUS.\n");
	if (intr_mask & 1 << 29)
		printk(KERN_INFO "[nvdebug] - Interrupt on XVE.\n");
	if (intr_mask & 1 << 30)
		printk(KERN_INFO "[nvdebug] - Interrupt on PRIV_RING.\n");
	if (intr_mask & 1 << 30)
		printk(KERN_INFO "[nvdebug] - Interrupt on SOFTWARE.\n");

	return IRQ_NONE; // We don't actually handle any interrupts. Pass them on.
}
#endif // INTERRUPT_DEBUG

// Find any and all NVIDIA GPUs in the system
// Note: This function fails if any of them are in a bad state
int probe_and_cache_devices(void) {
	// platform bus (SoC) iterators
	struct device *dev = NULL;
	struct device *temp_dev;
	// PCI search iterator and search query
	struct pci_dev *pcid = NULL;
	// This query pattern is mirrored off nouveau
	struct pci_device_id query = {
		.vendor = NV_PCI_VENDOR,  // Match NVIDIA devices
		.device = PCI_ANY_ID,
		.subvendor = PCI_ANY_ID,
		.subdevice = PCI_ANY_ID,
		.class_mask = 0xff << 16,
		.class = PCI_BASE_CLASS_DISPLAY << 16,  // Match display devs
	};
	int i = 0;
	// Search the platform bus for the first device that matches our name
	// Search for embedded GPU on Jetson (generic name starting around L4T 36.3)
	while (!dev && (temp_dev = bus_find_device_by_name(&platform_bus_type, dev, "17000000.gpu")))
		dev = temp_dev;
	// Search for GA10B (Jetson Orin)
	while (!dev && (temp_dev = bus_find_device_by_name(&platform_bus_type, dev, "17000000.ga10b")))
		dev = temp_dev;
	// Search for GV11B (Jetson Xavier)
	while (!dev && (temp_dev = bus_find_device_by_name(&platform_bus_type, dev, "17000000.gv11b")))
		dev = temp_dev;
	// Search for GP10B (Jetson TX2)
	while (!dev && (temp_dev = bus_find_device_by_name(&platform_bus_type, dev, "17000000.gp10b")))
		dev = temp_dev;
	// Search for GM20B (Jetson TX1)
	while (!dev && (temp_dev = bus_find_device_by_name(&platform_bus_type, dev, "57000000.gpu")))
		dev = temp_dev;
	// TODO: Support other platform bus devices (gk20a - TK1)
	if (dev) {
		mc_boot_0_t ids;
		struct platform_device *platd = container_of(dev, struct platform_device, dev);
		struct resource *regs = platform_get_resource(platd, IORESOURCE_MEM, 0);
		g_nvdebug_state[i].g = get_gk20a(dev);
		if (!regs)
			return -EADDRNOTAVAIL;
		g_nvdebug_state[i].regs = ioremap(regs->start, resource_size(regs));
		if (!g_nvdebug_state[i].regs) {
			printk(KERN_ERR "[nvdebug] Unable to map BAR0 on the integrated GPU\n");
			return -EADDRNOTAVAIL;
		}
		// The Jetson TX1, TX2, Xavier, and Orin do not have a BAR2 (but do have
		// BAR1). On the TX2+, all their platform resources are:
		//   [nvdebug] Region 0: Memory at 17000000 [size=16777216]
		//   [nvdebug] Region 1: Memory at 18000000 [size=16777216]
		//   [nvdebug] Region 2: Memory at 3b41000 [size=4096]
		// The TX1 has the same regions, but at different base addresses.
		g_nvdebug_state[i].bar3 = NULL;
		g_nvdebug_state[i].pcid = NULL;
		g_nvdebug_state[i].platd = platd;
		g_nvdebug_state[i].dev = dev;
		INIT_LIST_HEAD(&g_nvdebug_state[i].pd_allocs);
		// Don't check Chip ID until everything else is initalized
		ids.raw = nvdebug_readl(&g_nvdebug_state[i], NV_MC_BOOT_0);
		if (ids.raw == -1) {
			printk(KERN_ERR "[nvdebug] Unable to read config from Master Controller on the integrated GPU\n");
			return -EADDRNOTAVAIL;
		}
		g_nvdebug_state[i].chip_id = ids.chip_id;
		printk(KERN_INFO "[nvdebug] Chip ID %x (architecture %s) detected on platform bus and initialized.",
		       ids.chip_id, ARCH2NAME(ids.architecture));
		i++;
	}
	// Search the PCI bus and iterate through all matches
	// FIXME: Undo the pci_iomap() if this fails
	while ((pcid = pci_get_dev_by_id(&query, pcid)) && i < NVDEBUG_MAX_DEVICES) {
		mc_boot_0_t ids;
		g_nvdebug_state[i].g = NULL;
		// Map BAR0 (GPU control registers)
		// XXX: Don't use pci_iomap. This adds support for I/O registers, but we do
		//      not use the required ioread/write functions for those regions. We
		//      should use pci_ioremap_bar, which is explictly for MMIO regions.
		// pci_ioremap_bar -> ioremap_nocache (all platforms)
		// pci_iomap -> ioremap_nocache (on x86)
		g_nvdebug_state[i].regs = pci_iomap(pcid, 0, 0);
		if (!g_nvdebug_state[i].regs) {
			pci_err(pcid, "[nvdebug] Unable to map BAR0 on this GPU\n");
			return -EADDRNOTAVAIL;
		}
		// Map BAR3 (CPU-accessible mappings of GPU DRAM)
		g_nvdebug_state[i].bar3 = pci_iomap(pcid, 3, 0);
		// XXX: Try mapping only the lower half of BAR3 on fail
		// (vesafb may map the top half for display)
		if (!g_nvdebug_state[i].bar3)
			g_nvdebug_state[i].bar3 = pci_iomap(pcid, 3, pci_resource_len(pcid, 3)/2);
		// Observed on H100, BAR2, moved it BAR3, was moved to BAR4, and BAR1
		// was moved to BAR2.
		if (!g_nvdebug_state[i].bar3)
			g_nvdebug_state[i].bar3 = pci_iomap(pcid, 4, 0);
		g_nvdebug_state[i].pcid = pcid;
		g_nvdebug_state[i].platd = NULL;
		g_nvdebug_state[i].dev = &pcid->dev;
		INIT_LIST_HEAD(&g_nvdebug_state[i].pd_allocs);
		// Don't check Chip ID until everything else is initalized
		ids.raw = nvdebug_readl(&g_nvdebug_state[i], NV_MC_BOOT_0);
		if (ids.raw == -1) {
			pci_err(pcid, "[nvdebug] Unable to read config from Master Controller on this GPU\n");
			return -EADDRNOTAVAIL;
		}
		g_nvdebug_state[i].chip_id = ids.chip_id;
		printk(KERN_INFO "[nvdebug] Chip ID %x (architecture %s) detected on PCI bus and initialized.",
		       ids.chip_id, ARCH2NAME(ids.architecture));
#ifdef INTERRUPT_DEBUG
		// For this to work, you must also add IRQF_SHARED to the flags
		// argument of the request_threaded_irq() call in the nvidia driver
		// (file /usr/src/nvidia.../nvidia/nv.c and nv-msi.c with dkms)
		// Then run:
		//     sudo dkms remove nvidia-srv/VER -k $(uname -r)
		//     sudo dkms install nvidia-srv/VER -k $(uname -r) --force
		// where VER is the version of the nvidia module (eg. 535.216.03)
		int err;
		if ((err = request_irq(pcid->irq, nvdebug_irq_tap, IRQF_SHARED, "nvdebug tap", &g_nvdebug_state[i]))) {
			printk(KERN_WARNING "[nvdebug] Unable to initialize IRQ tap, error %d\n", err);
		}
#endif // INTERRUPT_DEBUG
		i++;
	}
	// Return the number of devices found
	if (i > 0)
		return i;
	return -ENODEV;
}

// Support: Fermi, Maxwell, Pascal, Volta, Turing
int get_last_runlist_id_gk104(struct nvdebug_state *g) {
	ptop_device_info_gk104_t info;
	int i, max_rl_id = 0; // Always at least one runlist
	// Figure out how many runlists there are by checking the device info
	// registers. Runlists are always numbered sequentially, so we just have
	// to find the highest-valued one and add 1 to get the number of runlists.
	for (i = 0; i < NV_PTOP_DEVICE_INFO__SIZE_1_GK104; i++) {
		if ((info.raw = nvdebug_readl(g, NV_PTOP_DEVICE_INFO_GK104(i))) == -1)
			return -EIO;
		if (info.info_type != INFO_TYPE_ENUM || !info.runlist_is_valid)
			continue;
		if (info.runlist_enum > max_rl_id)
			max_rl_id = info.runlist_enum;
	}
	return max_rl_id;
}

// Support: Ampere, Hopper, Ada (and newer likely)
// Identical structure to get_runlist_ram() in runlist.c. See comments there.
int get_last_runlist_id_ga100(struct nvdebug_state *g) {
	ptop_device_info_ga100_t ptop_entry;
	int i, runlist_count = 0;
	int ptop_size = NV_PTOP_DEVICE_INFO__SIZE_1_GA100(g);
	int ptop_entry_subrow = 0;
	for (i = 0; i < ptop_size; i++) {
		if ((ptop_entry.raw = nvdebug_readl(g, NV_PTOP_DEVICE_INFO_GA100(i))) == -1)
			return -EIO;
		if (!ptop_entry.raw)
			continue;
		if (ptop_entry_subrow == 2 && ptop_entry.rleng_id == 0)
			runlist_count++;
		if (ptop_entry.has_next_entry)
			ptop_entry_subrow += 1;
		else
			ptop_entry_subrow = 0;
	}
	return runlist_count - 1;
}

// Return the maximum runlist ID. For a two-runlist GPU, this would return 1.
int get_last_runlist_id(int device_id) {
	struct nvdebug_state* g = &g_nvdebug_state[device_id];
	if (g->chip_id >= NV_CHIP_ID_AMPERE)
		return get_last_runlist_id_ga100(g);
	else
		return get_last_runlist_id_gk104(g);
}

// Create files `/proc/gpu#/gpc#_tpc_mask`, world readable
// Support: Maxwell+
int create_tpc_mask_files(int device_id, struct proc_dir_entry *dir) {
	struct nvdebug_state* g = &g_nvdebug_state[device_id];
	char file_name[20];
	int i;
	struct proc_dir_entry *gpc_tpc_mask_entry;
	// Get maximum number of enabled GPCs for this chip
	uint32_t max_gpcs = nvdebug_readl(g, NV_PTOP_SCAL_NUM_GPCS);
	// Get a bitmask of which GPCs are disabled
	uint32_t gpcs_mask;
	if (g->chip_id < NV_CHIP_ID_AMPERE)
		gpcs_mask = nvdebug_readl(g, NV_FUSE_GPC_GM107);
	else
		gpcs_mask = nvdebug_readl(g, NV_FUSE_GPC_GA100);
	// Verify the reads succeeded
	if (max_gpcs == -1 || gpcs_mask == -1)
		return -EIO;
	// For each enabled GPC, expose a mask of disabled TPCs
	for (i = 0; i < max_gpcs; i++) {
		// Do nothing if GPC is disabled
		if ((1 << i) & gpcs_mask)
			continue;
		// If GPC is enabled, create an entry to read disabled TPCs mask
		snprintf(file_name, 20, "gpc%d_tpc_mask", i);
		if (g->chip_id < NV_CHIP_ID_AMPERE)
			gpc_tpc_mask_entry = proc_create_data(
				file_name, 0444, dir, compat_ops(&nvdebug_read_reg32_file_ops),
				(void*)(uintptr_t)NV_FUSE_TPC_FOR_GPC_GM107(i));
		else
			gpc_tpc_mask_entry = proc_create_data(
				file_name, 0444, dir, compat_ops(&nvdebug_read_reg32_file_ops),
				(void*)(uintptr_t)NV_FUSE_TPC_FOR_GPC_GA100(i));
		if (!gpc_tpc_mask_entry)
			return -ENOMEM;
	}
	return 0;
}

int __init nvdebug_init(void) {
	struct proc_dir_entry *dir;
	int err, res;
	// Check that an NVIDIA GPU is present and initialize g_nvdebug_state
	if ((res = probe_and_cache_devices()) < 0)
		return res;
	g_nvdebug_devices = res;
	// Create seperate ProcFS directories for each gpu
	while (res--) {
		uintptr_t last_runlist = 0;
		char device_id_str[7];
		// Create a wider copy of the GPU ID to allow us to abuse the *data
		// field of proc_dir_entry to store the GPU ID.
		uintptr_t device_id = res;
		// Create directory /proc/gpu# where # is the GPU number
		// As ProcFS entry creation only fails if out of memory, we auto-skip
		// to handling that on any error in creating ProcFS files.
		snprintf(device_id_str, 7, "gpu%ld", device_id);
		if (!(dir = proc_mkdir_data(device_id_str, 0555, NULL, (void*)device_id)))
			goto out_nomem;
		// Create files in the `/proc/gpu#/runlist#/` directory
		// The read handling code looks at the `pde_data` associated with the parent
		// directory to determine what the runlist ID is.
		if ((last_runlist = get_last_runlist_id(device_id)) < 0)
			return last_runlist;
		do {
			char runlist_name[12];
			struct proc_dir_entry *rl_dir;
			// Create `/proc/gpu#/runlist#` directory
			snprintf(runlist_name, 12, "runlist%lu", last_runlist);
			if (!(rl_dir = proc_mkdir_data(runlist_name, 0555, dir, (void*)device_id)))
				goto out_nomem;
			// Create one file for each runlist on Ampere+, or one file for each GPU on older
			if (g_nvdebug_state[res].chip_id >= NV_CHIP_ID_AMPERE || last_runlist == 0) {
				struct proc_dir_entry *chram_scope;
				// preempt_tsg, enable_channel, and disable_channel refer to a GPU-global channel
				// RAM on pre-Ampere GPUs
				if (g_nvdebug_state[res].chip_id >= NV_CHIP_ID_AMPERE)
					chram_scope = rl_dir;
				else
					chram_scope = dir;
				// Create file `/proc/gpu#/runlist#/preempt_tsg`, world writable
				// On Turing and older, `/proc/gpu#/preempt_tsg`
				if (!proc_create_data(
						"preempt_tsg", 0222, chram_scope, compat_ops(&preempt_tsg_file_ops),
						(void*)last_runlist))
					goto out_nomem;
				// Create file `/proc/gpu#/runlist#/disable_channel`, world writable
				// On Turing and older, `/proc/gpu#/disable_channel`
				if (!proc_create_data(
						"disable_channel", 0222, chram_scope, compat_ops(&disable_channel_file_ops),
						(void*)last_runlist))
					goto out_nomem;
				// Create file `/proc/gpu#/runlist#/enable_channel`, world writable
				// On Turing and older, `/proc/gpu#/enable_channel`
				if (!proc_create_data(
						"enable_channel", 0222, chram_scope, compat_ops(&enable_channel_file_ops),
						(void*)last_runlist))
					goto out_nomem;
				// Create file `/proc/gpu#/runlist#/wfi_preempt_channel`, world writable
				// On Turing and older, `/proc/gpu#/wfi_preempt_channel`
				if (!proc_create_data(
						"wfi_preempt_channel", 0222, chram_scope, compat_ops(&wfi_preempt_channel_file_ops),
						(void*)last_runlist))
					goto out_nomem;
				// Create file `/proc/gpu#/runlist#/cta_preempt_channel`, world writable
				// On Turing and older, `/proc/gpu#/cta_preempt_channel`
				if (!proc_create_data(
						"cta_preempt_channel", 0222, chram_scope, compat_ops(&cta_preempt_channel_file_ops),
						(void*)last_runlist))
					goto out_nomem;
				// Compute-instruction-level (CIL) preemption is only available on Pascal+
				if (g_nvdebug_state[res].chip_id >= NV_CHIP_ID_PASCAL) {
					// Create file `/proc/gpu#/runlist#/cil_preempt_channel`, world writable
					// On Turing and older, `/proc/gpu#/cil_preempt_channel`
					if (!proc_create_data(
							"cil_preempt_channel", 0222, chram_scope, compat_ops(&cil_preempt_channel_file_ops),
							(void*)last_runlist))
						goto out_nomem;
				}
				// Create files which enable on-GPU scheduling (Pascal+)
				if (g_nvdebug_state[res].chip_id >= NV_CHIP_ID_PASCAL) {
					// Create file `/proc/gpu#/map_mem_chid`, root writable
					if (!proc_create_data(
							"map_mem_chid", 0200, chram_scope, compat_ops(&map_mem_chid_file_ops),
							(void*)last_runlist))
						goto out_nomem;
					// Create file `/proc/gpu#/map_mem_ctxid`, root writable
					if (!proc_create_data(
							"map_mem_ctxid", 0222, rl_dir, compat_ops(&map_mem_ctxid_file_ops),
							(void*)last_runlist))
						goto out_nomem;
				}
			}
			// Create file `/proc/gpu#/runlist#/runlist`, world readable
			if (!proc_create_data(
					"runlist", 0444, rl_dir, compat_ops(&runlist_file_ops),
					(void*)last_runlist))
				goto out_nomem;
			// Create file `/proc/gpu#/runlist#/switch_to_tsg`, world writable
			if (!proc_create_data(
					"switch_to_tsg", 0222, rl_dir, compat_ops(&switch_to_tsg_file_ops),
					(void*)last_runlist))
				goto out_nomem;
			/* On the TU104, the context scheduler (contained in the Host, aka
			 * PFIFO, unit) has been observed to sometimes to fail to schedule TSGs
			 * containing re-enabled channels. Resubmitting the runlist
			 * configuration appears to remediate this condition, and so this API
			 * is exposed to help reset GPU scheduling as necessary.
			 */
			// Create file `/proc/gpu#/resubmit_runlist`, world writable
			if (!proc_create_data(
					"resubmit_runlist", 0222, rl_dir, compat_ops(&resubmit_runlist_file_ops),
					(void*)device_id))
				goto out_nomem;
		} while (last_runlist-- > 0);
		// Create file `/proc/gpu#/preempt_runlist`, world writable
		if (!proc_create_data(
				"preempt_runlist", 0222, dir, compat_ops(&preempt_runlist_file_ops),
				(void*)device_id))
			goto out_nomem;
		// Create file `/proc/gpu#/ack_bad_tsg`, world writable
		if (!proc_create_data(
				"ack_bad_tsg", 0222, dir, compat_ops(&ack_bad_tsg_file_ops),
				(void*)device_id))
			goto out_nomem;
		// Create file `/proc/gpu#/device_info`, world readable
		if (!proc_create_data(
				"device_info", 0444, dir, compat_ops(&device_info_file_ops),
				(void*)device_id))
			goto out_nomem;
		// Create file `/proc/gpu#/num_gpcs`, world readable
		if (!proc_create_data(
				"num_gpcs", 0444, dir, compat_ops(&nvdebug_read_reg32_file_ops),
				(void*)NV_PTOP_SCAL_NUM_GPCS))
			goto out_nomem;
		// Create file `/proc/gpu#/num_tpc_per_gpc`, world readable
		if (!proc_create_data(
				"num_tpc_per_gpc", 0444, dir, compat_ops(&nvdebug_read_reg32_file_ops),
				(void*)NV_PTOP_SCAL_NUM_TPC_PER_GPC))
			goto out_nomem;
		// Create file `/proc/gpu#/num_ces`, world readable
		if (!proc_create_data(
				"num_ces", 0444, dir, compat_ops(&nvdebug_read_reg32_file_ops),
				(void*)NV_PTOP_SCAL_NUM_CES))
			goto out_nomem;
		// Create files `/proc/gpu#/gpc#_tpc_mask`, world readable (Maxwell+)
		if (g_nvdebug_state[res].chip_id >= NV_CHIP_ID_MAXWELL)
			if ((err = create_tpc_mask_files(device_id, dir)))
				goto out_err;
		// Create file `/proc/gpu#/gpc_mask`, world readable (Maxwell+)
		if (g_nvdebug_state[res].chip_id >= NV_CHIP_ID_AMPERE) {
			if (!proc_create_data(
					"gpc_mask", 0444, dir, compat_ops(&nvdebug_read_reg32_file_ops),
					(void*)NV_FUSE_GPC_GA100))
				goto out_nomem;
		} else if (g_nvdebug_state[res].chip_id >= NV_CHIP_ID_MAXWELL) {
			if (!proc_create_data(
					"gpc_mask", 0444, dir, compat_ops(&nvdebug_read_reg32_file_ops),
					(void*)NV_FUSE_GPC_GM107))
				goto out_nomem;
		}
		// Create file `/proc/gpu#/CWD_SM_ID#`, world readable (Maxwell+)
		// Create file `/proc/gpu#/CWD_GPC_TPC_ID#`, world readable (Maxwell+)
		// - 6 entries on Maxwell (nvgpu)
		// - 16 entries on Pascal through Ampere (at least) (nvgpu, open-gpu-doc)
		// - 24 entries on Hopper through Ada (at least) (XXXX)
		// XXX: Only working while a context is active
		// XXX: Needed for libsmctrl2; hacky
		// Tested on GP104, TU102, GV100, AD102
		if (g_nvdebug_state[res].chip_id >= NV_CHIP_ID_HOPPER) {
			char file_name[21];
			long i;
			for (i = 0; i < 24; i++) {
				snprintf(file_name, 20, "CWD_SM_ID%ld", i);
				if (!proc_create_data(
						file_name, 0444, dir, compat_ops(&nvdebug_read_reg32_file_ops),
						(void*)(0x00405100+4*i))) // XXX: From XXXX
					goto out_nomem;
				// 18 entries on Ada (RTX 6000 Ada)
				// Returns 0xbadf1201 if GPU not active
				snprintf(file_name, 20, "CWD_GPC_TPC_ID%ld", i);
				if (!proc_create_data(
						file_name, 0444, dir, compat_ops(&nvdebug_read_reg32_file_ops),
						// Nothing between this location and CWD_SM_ID
						(void*)((0x00405000)+4*i))) // Found via reverse search from CWD_SM_ID location on Ada
					goto out_nomem;
				// Nothing in the following 28 words (before 0x00405220)
			}
		} else if (g_nvdebug_state[res].chip_id >= NV_CHIP_ID_MAXWELL) {
			char file_name[21];
			long i;
			union reg_range num_gpc_range;
			for (i = 0; i < 16; i++) {
				snprintf(file_name, 20, "CWD_SM_ID%ld", i);
				if (!proc_create_data(
						file_name, 0444, dir, compat_ops(&nvdebug_read_reg32_file_ops),
						(void*)(0x00405ba0+4*i))) // NV_PGRAPH_PRI_CWD_SM_ID(i)
					goto out_nomem;
				// ? entries on Maxwell
				// 8 entries on Pascal (test)
				// 16 entries on Volta through Ampere (open-gpu-doc)
				// Returns 0 if GPU ont active
				snprintf(file_name, 20, "CWD_GPC_TPC_ID%ld", i);
				if (!proc_create_data(
						file_name, 0444, dir, compat_ops(&nvdebug_read_reg32_file_ops),
						(void*)(0x00405b60+4*i))) // NV_PGRAPH_PRI_CWD_GPC_TPC_ID(i)
					goto out_nomem;
			}
			num_gpc_range.offset = 0x00405b00; // NV_PGRAPH_PRI_CWD_FS
			// Lower eight bits of register are _NUM_GPCS
			num_gpc_range.start_bit = 0;
			num_gpc_range.stop_bit = 8;
			if (!proc_create_data(
					"CWD_FS_NUM_GPCS", 0444, dir, compat_ops(&nvdebug_read_reg_range_file_ops),
					(void*)(num_gpc_range.raw)))
				goto out_nomem;
			num_gpc_range.start_bit = 8;
			num_gpc_range.stop_bit = 16;
			if (!proc_create_data(
					"CWD_FS_NUM_TPCS", 0444, dir, compat_ops(&nvdebug_read_reg_range_file_ops),
					(void*)(num_gpc_range.raw)))
				goto out_nomem;
		}
		// Create file `/proc/gpu#/local_memory`, world readable (Pascal+)
		if (g_nvdebug_state[res].chip_id >= NV_CHIP_ID_PASCAL) {
			if (!proc_create_data(
					"local_memory", 0444, dir, compat_ops(&local_memory_file_ops),
					(void*)0x00100ce0))
				goto out_nomem;
		}
		// Create files exposing LCE and PCE configuration (Pascal+)
		if (g_nvdebug_state[res].chip_id >= NV_CHIP_ID_PASCAL) {
			// Create file `/proc/gpu#/copy_topology`, world readable
			if (!proc_create_data(
					"copy_topology", 0444, dir, compat_ops(&copy_topology_file_ops),
					(void*)0))
				goto out_nomem;
			// Create file `/proc/gpu#/pce_map`, world readable
			if (!proc_create_data(
					"pce_map", 0444, dir, compat_ops(&nvdebug_read_reg32_file_ops),
					(void*)NV_CE_PCE_MAP))
				goto out_nomem;
		}
		// Create files exposing subcontext partitioning (Volta+)
		// TODO: Make this not a hack with undocumented magic numbers
		if (g_nvdebug_state[res].chip_id >= NV_CHIP_ID_VOLTA) {
			char file_name[21];
			long i;
			// Create file `/proc/gpu#/partition_ctl`, world readable
			if (!proc_create_data(
					"partition_ctl", 0444, dir, compat_ops(&nvdebug_read_reg32_file_ops),
					(void*)0x00405b2c))
				goto out_nomem;
			// Create file `/proc/gpu#/partition_data`, world readable
			if (!proc_create_data(
					"partition_data", 0444, dir, compat_ops(&nvdebug_read_reg32_file_ops),
					(void*)0x00405b30))
				goto out_nomem;
			// Create file `/proc/gpu#/partition_data#`, world readable
			for (i = 0; i < 64; i++) {
				snprintf(file_name, 20, "partition_data%ld", i);
				if (!proc_create_data(
						file_name, 0444, dir, compat_ops(&nvdebug_read_part_file_ops),
						(void*)i))
					goto out_nomem;
			}
			// For debugging what MPS is changing
			// Create file `/proc/gpu#/CWD_CG0`, world readable
			if (!proc_create_data(
					"CWD_CG0", 0444, dir, compat_ops(&nvdebug_read_reg32_file_ops),
					(void*)0x00405bf0))
				goto out_nomem;
			// Create file `/proc/gpu#/CWD_CG1`, world readable
			if (!proc_create_data(
					"CWD_CG1", 0444, dir, compat_ops(&nvdebug_read_reg32_file_ops),
					(void*)0x00405bf4))
				goto out_nomem;
			// Create file `/proc/gpu#/CWD_GPC_TPC_ID#`, world readable
			// This does not appear to work on Hopper. Works on Ampere.
			/*for (i = 0; i < 16; i++) {
				snprintf(file_name, 20, "CWD_GPC_TPC_ID%ld", i);
				if (!proc_create_data(
						file_name, 0444, dir, compat_ops(&nvdebug_read_reg32_file_ops),
						(void*)(0x00405b60+4*i)))
					goto out_nomem;
			}*/
		}
	}
	// (See Makefile if you want to know the origin of GIT_HASH.)
	printk(KERN_INFO "[nvdebug] Module version "GIT_HASH" initialized\n");
	return 0;
out_nomem:
	err = -ENOMEM;
out_err:
	// Make sure to clear all ProcFS directories on error
	while (res < g_nvdebug_devices) {
		char device_id_str[7];
		snprintf(device_id_str, 7, "gpu%d", res);
		remove_proc_subtree(device_id_str, NULL);
		res++;
	}
	return err;
}

static void __exit nvdebug_exit(void) {
	struct nvdebug_state *g;
	// Deinitialize each device
	while (g_nvdebug_devices--) {
		// Remove procfs directory
		char device_id[7];
		snprintf(device_id, 7, "gpu%d", g_nvdebug_devices);
		remove_proc_subtree(device_id, NULL);
		// Force-free associated allocations
		g = &g_nvdebug_state[g_nvdebug_devices];
		gc_page_directory(g, true);
		// Free BAR mappings for PCIe devices
		if (g && g->pcid) {
#ifdef INTERRUPT_DEBUG
			// IRQ handler uses g->regs, so free IRQ first
			free_irq(g->pcid->irq, g);
#endif // INTERRUPT_DEBUG
			if (g->regs)
				pci_iounmap(g->pcid, g->regs);
			if (g->bar2)
				pci_iounmap(g->pcid, g->bar2);
		} else {
			if (g->regs)
				iounmap(g->regs);
		}
		printk(KERN_INFO "[nvdebug] Chip ID %x deinitialized.", g->chip_id);
	}
	printk(KERN_INFO "[nvdebug] Module exit complete.\n");
}

module_init(nvdebug_init);
module_exit(nvdebug_exit);
