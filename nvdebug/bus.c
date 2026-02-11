/* Copyright 2024 Joshua Bakita
 * Helpers for dealing with the PCIe and platform bus
 *
 * = Design Notes =
 * We have to use PRAMIN to access the BAR2 page table. While it's typically also
 * mapped into BAR2, we have no way to know where without the table. If the table
 * changes, the new sections will have new mappings in BAR2, repeating the
 * problem, and making caching insufficient.
 *
 * = Terms =
 * VRAM/VID_MEM: Video RAM; Addresses to physical frames in the on-GPU RAM.
 * SYS_MEM: System Memory; "Bus addresses"; Addresses which can be presented to
 *          the PCIe Host for resolution. On x86_64 without an IOMMU, these are
 *          just physical addresses, but may be I/O Virtual Addresses (IOVAs)
 *          or translated via an I/O MMU on other platforms; DMA Addresses.
 * PEER: Addresses to RAM on another GPU.
 */
#include <linux/printk.h> // For printk()
#include <asm/errno.h> // For error defines
#include <asm/io.h> // For readl()

#include "nvdebug.h"

/* Obtain the PRAMIN offset at which `addr` can be accessed
  @param addr   Address to find
  @param target Which address space to use (VRAM, SYS_MEM, PEER(?))
  @return positive offset, -EINVAL on invalid arguments, or -EOPNOTSUPP on
          an unsupported platform.

  Note: Will move the PRAMIN window to accomodate the request. Only guarantees
        that the surrounding 64-KiB-aligned window will be accessible.
  Note: Moving the PRAMIN window will cause problems if it races with driver
        code that tries to do the same, or expects the window not to move.
  Bugs: Untested on PEER.
*/
int addr_to_pramin_mut(struct nvdebug_state *g,
                       uint64_t addr, enum INST_TARGET target) {
	bar0_window_t window;
	uint64_t pramin_base;
	uint32_t window_reg;
	// For us, accuracy and robustness is more important than speed
	// Check that the address is valid (49 bits are addressable on-GPU, but
	// PRAMIN only supports up to 40 bits).
	if (addr & ~0x000000ffffffffff) {
		printk(KERN_ERR "[nvdebug] Invalid address %llx passed to %s!\n",
		       addr, __func__);
		return -EINVAL;
	}
	// Register relocated on Hopper and Blackwell+
	if ((g->chip_id >= NV_CHIP_ID_HOPPER && g->chip_id < NV_CHIP_ID_ADA) || g->chip_id >= NV_CHIP_ID_BLACKWELL)
		window_reg = NV_XAL_EP_BAR0_WINDOW_BASE;
	else
		window_reg = NV_PBUS_BAR0_WINDOW;
	if ((window.raw = nvdebug_readl(g, window_reg)) == -1) {
		printk(KERN_ERR "[nvdebug] PRAMIN window configuration inaccessible; "
		       "failing %s\n", __func__);
		return -EOPNOTSUPP;
	}
	if (window.target != target) {
		// On Hopper and Blackwell+, the window always points at VID_MEM
		if ((g->chip_id >= NV_CHIP_ID_HOPPER && g->chip_id < NV_CHIP_ID_ADA) || g->chip_id >= NV_CHIP_ID_BLACKWELL)
			return -EOPNOTSUPP;
		else
			goto relocate;
	}
	pramin_base = ((uint64_t)window.base) << 16;
	if (addr < pramin_base || addr >= pramin_base + NV_PRAMIN_LEN)
		goto relocate;
	return addr - pramin_base; // Guaranteed to be < 1MiB, so safe for int
relocate:
	printk(KERN_INFO "[nvdebug] [SIDE EFFECT] Moving PRAMIN window from base "
	       "%llx (%s) to %llx (%s) to accomodate %#018llx\n",
	       ((uint64_t)window.base) << 16, target_to_text(window.target),
	       (addr >> 16) << 16, target_to_text(target), addr);
	// Move PRAMIN window to a 64KiB-aligned address
	window.base = (u32)(addr >> 16); // Safe, due to above range check
	window.target = target;
	nvdebug_writel(g, window_reg, window.raw);
	// Wait for the window to move by re-reading (as done in nvgpu driver)
	(void) nvdebug_readl(g, window_reg);
	return (int)(addr & 0xffffull);
}

/* Get a copy of the BAR2 page directory configuration (base and aperture)
  @param pd Pointer at which to store the configuration, including a pointer
            and aperture for the zeroth entry of the top-level page directory
            (PD3 for V2 page tables). This pointer **may not** be directly
            dereferencable, and the caller may need to shift the BAR2 window.
  @return 0 on success, -errno on error.
  Note: This may move the PRAMIN window.
*/
int get_bar2_pdb(struct nvdebug_state *g, page_dir_config_t* pd) {
	int ret;
	uint64_t bar2_ptr;
	enum INST_TARGET bar2_target;
	bool bar2_is_virtual;

	if (!pd)
		return -EINVAL;

	if (!g->bar2)
		return -ENXIO;

	// BAR2 has its own instance block (typically in VRAM) which contains the
	// Page Directory Base (PDB), a pointer to a page directory/table
	// hierarchy used to translate BAR2 offsets to VRAM or SYS_MEM addresses.

	// Determine location of BAR2 instance block
	if ((g->chip_id >= NV_CHIP_ID_HOPPER && g->chip_id < NV_CHIP_ID_ADA) || g->chip_id >= NV_CHIP_ID_BLACKWELL) {
		// Register layout updated on Hopper and Blackwell+ to support 52-bit
		// instance block pointers (vs. 40 bits before)
		bar_config_block_gh100_t bar2_block;
		if ((bar2_block.raw = nvdebug_readq(g, NV_VIRTUAL_FUNCTION_PRIV_FUNC_BAR2_BLOCK)) == -1) {
			printk(KERN_ERR "[nvdebug] Unable to read BAR2/3 configuration! BAR2/3 inaccessible.\n");
			return -EOPNOTSUPP;
		}
		bar2_ptr = (uint64_t)bar2_block.ptr << 12;
		bar2_target = bar2_block.target;
		bar2_is_virtual = bar2_block.is_virtual;
	} else {
		bar_config_block_t bar2_block;
		if ((bar2_block.raw = nvdebug_readl(g, NV_PBUS_BAR2_BLOCK)) == -1) {
			printk(KERN_ERR "[nvdebug] Unable to read BAR2/3 configuration! BAR2/3 inaccessible.\n");
			return -EOPNOTSUPP;
		}
		bar2_ptr = (uint64_t)bar2_block.ptr << 12;
		bar2_target = bar2_block.target;
		bar2_is_virtual = bar2_block.is_virtual;
	}
	printk(KERN_INFO "[nvdebug] BAR2 inst block @ %llx in %s's %s address space.\n", bar2_ptr, target_to_text(bar2_target), bar2_is_virtual ? "virtual" : "physical");
	// Setup PRAMIN to point at the BAR2 instance block
	// TODO: This won't work if the instance block is in SYS_MEM on Hopper or
	//       Blackwell+. Going through the I/O MMU appears to be fairly
	//       reliable, so I need to switch to using that logic whenever
	//       SYS_MEM may be accessed.
	if ((ret = addr_to_pramin_mut(g, bar2_ptr, bar2_target)) < 0) {
		printk(KERN_ERR "[nvdebug] Unable to access BAR2/3 Instance Block configuration via PRAMIN! BAR2/3 inaccessible.\n");
		return ret;
	}
	// Pull the page directory base configuration from the instance block
	if ((pd->raw = nvdebug_readq(g, NV_PRAMIN + ret + NV_PRAMIN_PDB_CONFIG_OFF)) == -1) {
		printk(KERN_ERR "[nvdebug] Unable to read BAR2/3 PDB configuration! BAR2/3 inaccessible.\n");
		return -EOPNOTSUPP;
	}

	return 0;
}
