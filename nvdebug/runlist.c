/* Copyright 2024 Joshua Bakita
 * Helpers for dealing with the runlist and other Host (PFIFO) registers
 */
#include <linux/iommu.h>  // iommu_get_domain_for_dev() and iommu_iova_to_phys()
#include <linux/printk.h> // For printk()
#include <asm/errno.h> // For error defines
#include <asm/io.h> // For phys_to_virt()

#include "nvdebug.h"

/* Get RunList RAM (RLRAM) offset for a runlist from the device topology
  @param rl_id      Which runlist to obtain [numbered in order of appearance in
                    the device topology (PTOP) registers]
  @param rl_ram_off Location at which to store runlist private register
                    interface base address (PRI base); an offset into the BAR0
                    register range.
  @return 0 or -errno on error
*/
int get_runlist_ram(struct nvdebug_state *g, int rl_id, uint32_t *rl_ram_off) {
	int i;
	int curr_rl_id = 0;
	int ptop_size = NV_PTOP_DEVICE_INFO__SIZE_1_GA100(g);
	// Each PTOP entry is composed of 1--3 subrows, and the fields available
	// on each row vary. The runlist RAM location is only available on row 3
	int ptop_entry_subrow = 0;
	ptop_device_info_ga100_t ptop_entry;
	// Iterate through all PTOP entries
	for (i = 0; i < ptop_size; i++) {
		if ((ptop_entry.raw = nvdebug_readl(g, NV_PTOP_DEVICE_INFO_GA100(i))) == -1)
			return -EIO;
		// Skip empty entries
		if (!ptop_entry.raw)
			continue;
		// If on subrow 3 (zero-base-index 2), runlist info is available
		// Multiple engines may be associated with a single runlist, so
		// multiple PTOP entries may refer to the same runlist. Only match when
		// on the 0th-associated entry.
		if (ptop_entry_subrow == 2 && ptop_entry.rleng_id == 0) {
			// If this is the requested runlist, return it
			if (curr_rl_id == rl_id) {
				*rl_ram_off = (uint32_t)ptop_entry.runlist_pri_base << 10;
				return 0;
			}
			// Otherwise, update our accounting of what the next runlist ID is
			curr_rl_id++;
		}
		// Track if the next row is a subrow of the current entry
		if (ptop_entry.has_next_entry)
			ptop_entry_subrow += 1;
		else
			ptop_entry_subrow = 0;
	}
	// Search failed; requested index does not exist
	return -EINVAL;
}

/* Get runlist head and info (incl. length)
  @param rl_id   Which runlist to obtain?
  @param rl_iter Location at which to store output
  @return 0 or -errno on error
*/
int get_runlist_iter(struct nvdebug_state *g, int rl_id, struct runlist_iter *rl_iter) {
	uint64_t runlist_iova;
	enum INST_TARGET runlist_target;
	uint16_t runlist_len;
	int err;
#ifdef FALLBACK_TO_PRAMIN
	int off;
#endif // FALLBACK_TO_PRAMIN
	// Zero-initialize the runlist iterator
	*rl_iter = (struct runlist_iter){0};

	// Get runlist location and length using architecture-dependent logic
	if (g->chip_id < NV_CHIP_ID_TURING) {
		eng_runlist_gf100_t rl;
		if ((rl.raw = nvdebug_readq(g, NV_PFIFO_ENG_RUNLIST_BASE_GF100(rl_id))) == -1)
			return -EIO;
		runlist_iova = ((uint64_t)rl.ptr) << 12;
		runlist_target = rl.target;
		runlist_len = rl.len;
		printk(KERN_INFO "[nvdebug] Runlist %d for %x: %d entries @ %llx in %s (config raw: %#018llx)\n",
		       rl_id, g->chip_id, rl.len, runlist_iova, target_to_text(rl.target), rl.raw);
	} else if (g->chip_id < NV_CHIP_ID_AMPERE) {
		runlist_base_tu102_t base;
		runlist_submit_tu102_t submit;
		if ((base.raw = nvdebug_readq(g, NV_PFIFO_RUNLIST_BASE_TU102(rl_id))) == -1)
			return -EIO;
		if ((submit.raw = nvdebug_readq(g, NV_PFIFO_RUNLIST_SUBMIT_TU102(rl_id))) == -1)
			return -EIO;
		runlist_iova = ((uint64_t)base.ptr) << 12;
		runlist_target = base.target;
		runlist_len = submit.len;
		printk(KERN_INFO "[nvdebug] Runlist %d for %x: %d entries @ %llx in %s (config raw: %#018llx %#018llx)\n",
		       rl_id, g->chip_id, submit.len, runlist_iova, target_to_text(runlist_target), base.raw, submit.raw);
	} else {
		runlist_base_tu102_t base;
		runlist_submit_tu102_t submit;
		uint32_t runlist_pri_base;
		// Runlist configurations are stored in per-runlist regions on Ampere+
		if ((err = get_runlist_ram(g, rl_id, &runlist_pri_base)) < 0)
			return err;
		// The runlist configuration region (RLRAM) contains Turing-like BASE
		// and SUBMIT registers at static offsets
		if ((base.raw = nvdebug_readq(g, runlist_pri_base + NV_RUNLIST_BASE_GA100)) == -1)
			return -EIO;
		if ((submit.raw = nvdebug_readq(g, runlist_pri_base + NV_RUNLIST_SUBMIT_GA100)) == -1)
			return -EIO;
		runlist_iova = ((uint64_t)base.ptr) << 12;
		runlist_target = base.target;
		runlist_len = submit.len;
		printk(KERN_INFO "[nvdebug] Runlist %d for %x: %d entries @ %llx in %s (config raw: %#018llx %#018llx)\n",
		       rl_id, g->chip_id, submit.len, runlist_iova, target_to_text(runlist_target), base.raw, submit.raw);
		printk(KERN_INFO "[nvdebug] Runlist offset is %d\n", submit.offset);
		rl_iter->runlist_pri_base = runlist_pri_base;
	}
	// Return early on an empty runlist
	if (!runlist_len)
		return 0;

	// If the runlist is in VID_MEM, search the BAR2/3 page tables for a mapping
	if (runlist_target == TARGET_VID_MEM) {
		uint64_t runlist_bar_vaddr;
		page_dir_config_t pd_config;

		if ((err = get_bar2_pdb(g, &pd_config)) < 0)
			goto attempt_pramin_access;

		// XXX: PD version detection not working on Hopper [is_ver2 errantly (?) unset]
		if (g->chip_id >= NV_CHIP_ID_HOPPER && g->chip_id < NV_CHIP_ID_ADA) {
			printk(KERN_WARNING "[nvdebug] V3 page tables do not currently work on Hopper! Mystery config: %llx\n", pd_config.raw);
			err = -EOPNOTSUPP;
			goto attempt_pramin_access;
		}
		if (pd_config.is_ver2)
			runlist_bar_vaddr = search_page_directory(g, pd_config, runlist_iova, TARGET_VID_MEM);
		else
			runlist_bar_vaddr = search_v1_page_directory(g, pd_config, runlist_iova, TARGET_VID_MEM);
		if (!runlist_bar_vaddr) {
			printk(KERN_WARNING "[nvdebug] Unable to find runlist %d mapping in BAR2/3 page tables for %x.\n", rl_id, g->chip_id);
			err = -EOPNOTSUPP;
			goto attempt_pramin_access;
		}

		printk(KERN_INFO "[nvdebug] Runlist %d for %x @ %llx in BAR2 virtual address space.\n", rl_id, g->chip_id, runlist_bar_vaddr);
		if (!g->bar2) {
			printk(KERN_WARNING "[nvdebug] BAR2/3 not mapped for %x.\n", g->chip_id);
			return -ENODEV;
		}
		rl_iter->curr_entry = g->bar2 + runlist_bar_vaddr;
	} else {
		// Directly access the runlist if stored in SYS_MEM (physically addressed)
		// XXX: SYS_MEM is an IOMMU address on some platforms, causing this to crash
		rl_iter->curr_entry = (void*)phys_to_virt(runlist_iova);
	}
	rl_iter->len = runlist_len;
	return 0;

attempt_pramin_access:
#ifdef FALLBACK_TO_PRAMIN
	printk(KERN_INFO "[nvdebug] Attempting to move PRAMIN window to runlist as BAR2/3-based access failed [DANGEROUS SIDE EFFECTS]!\n");
	if ((off = addr_to_pramin_mut(g, runlist_iova, runlist_target)) == -1)
		return off;
	rl_iter->curr_entry = g->regs + NV_PRAMIN + off;
	rl_iter->len = runlist_len;
	return 0;
#else
	return err;
#endif // FALLBACK_TO_PRAMIN
}

/* Trigger a preempt of the specified TSG
  @param tsg_id ID of TSG to preempt.
  @param rl_id  Which channel RAM address space to search?
  @return 0 or -errno on error

  Note: If no other TSGs exist in the associated runlist, this TSG may
        continue executing, unless NV_PFIFO_SCHED_DISABLE is set, or all the
        channels of the TSG to be preempted are disabled.
*/
int preempt_tsg(struct nvdebug_state *g, uint32_t rl_id, uint32_t tsg_id) {
	pfifo_preempt_t preempt;
	// Fermi does not support time-slice groups
	if (g->chip_id < NV_CHIP_ID_KEPLER)
		return -EOPNOTSUPP;

	preempt.raw = 0;
	preempt.id = tsg_id;
	preempt.type = PREEMPT_TYPE_TSG;

	// Actually trigger the preemption
	if (g->chip_id < NV_CHIP_ID_AMPERE) {
		nvdebug_writel(g, NV_PFIFO_PREEMPT, preempt.raw);
	} else {
		uint32_t runlist_reg_base;
		int err;
		// As TSG and channel IDs are namespaced per-runlist starting with
		// Ampere, the PREEMPT register is also per-runlist.
		if ((err = get_runlist_ram(g, rl_id, &runlist_reg_base)))
			return err;
		nvdebug_writel(g, runlist_reg_base + NV_RUNLIST_PREEMPT_GA100, preempt.raw);
	}
	return 0;
}

/* Trigger a preempt of the specified runlist
  @param rl_id ID of runlist to preempt.
  @return 0 or -errno on error
*/
int preempt_runlist(struct nvdebug_state *g, uint32_t rl_id) {
	// The runlist preempt register does not exist on Kepler (tested gk104)
	if (g->chip_id < NV_CHIP_ID_MAXWELL)
		return -EOPNOTSUPP;

	// Write to trigger the preemption (the register contains nothing to
	// preserve, and can thus just be overwritten)
	if (g->chip_id < NV_CHIP_ID_AMPERE) {
		runlist_preempt_t rl_preempt;
		rl_preempt.raw = BIT(rl_id);
		nvdebug_writel(g, NV_PFIFO_RUNLIST_PREEMPT, rl_preempt.raw);
	} else {
		int err;
		uint32_t runlist_regs_base;
		pfifo_preempt_t preempt;
		// The RUNLIST_PREEMPT register was deleted, and the _PREEMPT register
		// was extended to support runlist-level preemptions starting on Ampere
		preempt.id = rl_id;
		preempt.type = PREEMPT_TYPE_RUNLIST;
		// The preempt register is scoped per-runlist on Ampere+
		if ((err = get_runlist_ram(g, rl_id, &runlist_regs_base)))
			return err;
		nvdebug_writel(g, runlist_regs_base + NV_RUNLIST_PREEMPT_GA100, preempt.raw);
	}
	return 0;
}

// Read and write runlist configuration, triggering a resubmit
int resubmit_runlist(struct nvdebug_state *g, uint32_t rl_id, uint32_t off) {
	// Necessary registers do not exist pre-Fermi
	if (g->chip_id < NV_CHIP_ID_FERMI)
		return -EOPNOTSUPP;

	if (g->chip_id < NV_CHIP_ID_TURING) {
		eng_runlist_gf100_t rl;
		if (rl_id > MAX_RUNLISTS_GF100)
			return -EINVAL;
		if ((rl.raw = nvdebug_readq(g, NV_PFIFO_ENG_RUNLIST_BASE_GF100(rl_id))) == -1)
			return -EIO;
		rl.id = rl_id;
		nvdebug_writeq(g, NV_PFIFO_RUNLIST_BASE_GF100, rl.raw);
	} else if (g->chip_id < NV_CHIP_ID_AMPERE) {
		runlist_submit_tu102_t submit;
		if (rl_id > MAX_RUNLISTS_TU102)
			return -EINVAL;
		if ((submit.raw = nvdebug_readq(g, NV_PFIFO_RUNLIST_SUBMIT_TU102(rl_id))) == -1)
			return -EIO;
		preempt_runlist(g, rl_id);
		if (off != -1)
			submit.offset = off;
		nvdebug_writeq(g, NV_PFIFO_RUNLIST_SUBMIT_TU102(rl_id), submit.raw);
	} else {
		int err;
		uint32_t runlist_pri_base;
		runlist_submit_tu102_t submit;
		if ((err = get_runlist_ram(g, rl_id, &runlist_pri_base)) < 0)
			return err;
		if ((submit.raw = nvdebug_readq(g, runlist_pri_base + NV_RUNLIST_SUBMIT_GA100)) == -1)
			return -EIO;
		preempt_runlist(g, rl_id);
		if (off != -1)
			submit.offset = off;
		// On Ampere, this does not appear to trigger a preempt of the
		// currently-running channel (even if the currently running channel
		// becomes disabled), but will cause newly re-enabled channels
		// (at least if nothing else is pending) to become ready (tested on
		// Jetson Orin).
		nvdebug_writeq(g, runlist_pri_base + NV_RUNLIST_SUBMIT_GA100, submit.raw);
	}
	return 0;
}

/* Get a CPU-accessible pointer to an arbitrary-address-space instance block
  @param instance_addr  Address of instance block
  @param intasce_target Aperture/taget of instance block address
  @return A dereferencable KVA, NULL if not found, or an ERR_PTR-wrapped error

  Note: The returned address will be a BAR2 or physical address, mapped into
        kernel space, /not/ a PRAMIN-derived address. Thus, the returned
        address will have an indefinite lifetime, and will be uneffected by use
        of PRAMIN elsewhere (such as to read the CTXSW block).
*/
instance_ctrl_t *instance_deref(struct nvdebug_state *g, uint64_t instance_addr,
                                enum INST_TARGET instance_target) {
	if (!instance_addr || instance_target == TARGET_INVALID)
		return ERR_PTR(-EINVAL);
	if (instance_target == TARGET_VID_MEM) {
		int err;
		uint64_t inst_bar_vaddr;
		page_dir_config_t pd_config;
		// Only access VID_MEM via BAR2; do not fall back to PRAMIN
		if (!g->bar2)
			return NULL;
		// Find page tables which define how BAR2/3 offsets are translated to
		// physical VID/SYS_MEM addresses.
		if ((err = get_bar2_pdb(g, &pd_config)) < 0) {
			printk(KERN_ERR "[nvdebug] Error: Unable to access page directory "
			       "configuration for BAR2/3. Error %d.\n", err);
			return ERR_PTR(err);
		}
		// Search the BAR2/3 page tables for the offset at which the instance
		// block is mapped (reverse translation).
		if (pd_config.is_ver2)
			inst_bar_vaddr = search_page_directory(g, pd_config, instance_addr, instance_target);
		else
			inst_bar_vaddr = search_v1_page_directory(g, pd_config, instance_addr, instance_target);
		if (!inst_bar_vaddr) {
			printk(KERN_WARNING "[nvdebug] Warning: Instance block %#018llx "
			       "(%s) appears unmapped in BAR2/3.\n", instance_addr,
			       target_to_text(instance_target));
			return NULL;
		}
		return g->bar2 + inst_bar_vaddr;
	} else {
		struct iommu_domain *dom;
		// SYS_MEM addresses are physical addresses *from the perspective of
		// the device* ("bus addresses"), and may not necessarially correspond
		// to physical addresses from the perspective of the CPU. The I/O MMU
		// is responsible for mapping bus addresses to CPU-relative physical
		// addresses when there is no direct correspondence. If an I/O MMU is
		// enabled on this GPU, ask it to translate the bus address to a
		// CPU-relative physical address.
		if ((dom = iommu_get_domain_for_dev(g->dev))) {
			// XXX: As of Aug 2024, this is not tested, so include extra logging
			printk(KERN_DEBUG "[nvdebug] I/O MMU translated SYS_MEM I/O VA %#llx for instance block", instance_addr);
			if (!(instance_addr = iommu_iova_to_phys(dom, instance_addr))) {
				printk(KERN_ERR "[nvdebug] Error: I/O MMU failed to translate "
				       "%#018llx (%s) to a CPU-relative physical address.\n",
				       instance_addr, target_to_text(instance_target));
				return ERR_PTR(-EADDRNOTAVAIL);
			}
			printk(KERN_DEBUG " to physical address %#llx.\n", instance_addr);
		}
		// Convert from a physical address to a kernel virtual address (KVA)
		return phys_to_virt(instance_addr);
	}
}

/* Get a CPU-accessible pointer to the CTXSW block for a channel intance block
  @param inst Dereferencable pointer to the start of a complete instance block
  @return A dereferencable KVA, NULL if not found, or an ERR_PTR-wrapped error

  Note: The returned address **will** be a PRAMIN-based address. Any changes to
        PRAMIN **will** invalidate the returned pointer. `inst` **cannot** be a
        pointer into the PRAMIN space.
*/
context_switch_ctrl_t *get_ctxsw(struct nvdebug_state *g,
                                 instance_ctrl_t *inst) {
	int err;
	context_switch_ctrl_t *wfi = NULL;
	uint64_t wfi_virt, wfi_phys, ctxsw_virt, ctxsw_phys;
	enum INST_TARGET wfi_phys_aperture, ctxsw_phys_aperture;

	// The WFI block contains a pointer to the CTXSW block, which contains the
	// preemption mode configuration for the context. (As best I can tell, the WFI
	// block is subcontext-specific, whereas the CTXSW block is context-wide.
	wfi_virt = (uint64_t)inst->engine_wfi_ptr << 12;

	// WFI may not be configured
	if (!wfi_virt)
		goto out;

	// Determine the physical location of the WFI block
	if (inst->engine_wfi_is_virtual) {
		if (inst->pdb.is_ver2)
			err = translate_page_directory(g, inst->pdb, wfi_virt, &wfi_phys, &wfi_phys_aperture);
		else
			err = translate_v1_page_directory(g, inst->pdb, wfi_virt, &wfi_phys, &wfi_phys_aperture);
		if (err) {
			printk(KERN_ERR "[nvdebug] Critical: Inconsistent GPU state; WFI block "
			       "pointer %#018llx (virt) cannot be found in process page tables! "
			       "Translation error %d.\n", wfi_virt, -err);
			return ERR_PTR(-ENOTRECOVERABLE);
		}
	} else {
		wfi_phys = (uint64_t)inst->engine_wfi_ptr << 12;
		wfi_phys_aperture = inst->engine_wfi_target;
	}

	// Get a dereferencible pointer to the WFI block (the WFI and CTXSW blocks
	// have not been observed as mapped in BAR2/3, so we use the PRAMIN window).
	// Note: On Jetson boards, we could attempt to avoid PRAMIN since CTXSW is in
	//       SYS_MEM, but this function will always need to use PRAMIN to work
	//       around the WFI and CTXSW blocks not being accessible via BAR2/3 on
	//       PCIe GPU, so always use PRAMIN for simplicity.
	if ((wfi_phys = addr_to_pramin_mut(g, wfi_phys, wfi_phys_aperture)) == -1)
		goto out;
	wfi = g->regs + wfi_phys + NV_PRAMIN;

// XXX
//	return wfi;
// End XXX

	// While the WFI block uses the same layout as the context switch (CTXSW)
	// control block, it is mostly unpopulated except for a few pointers on GPUs
	// after Volta. This appears to be related to subcontexts, where each
	// subcontext has its own WFI block containing a pointer to the overarching
	// CTXSW block. Only attempt to find the overarching CTXSW block if at least
	// one subcontext is enabled.
	if (inst->subcontext_pdb_valid) {
		// Subcontexts are Volta+-only. Volta only supports Page Table Ver. 2
		if (!inst->pdb.is_ver2)
			return ERR_PTR(-ENOTRECOVERABLE);
		// Obtain the address of the CXTSW block in this context
		ctxsw_virt = wfi->context_buffer_ptr_hi;
		ctxsw_virt <<= 32;
		ctxsw_virt |= wfi->context_buffer_ptr_lo;
		if (!ctxsw_virt) {
			printk(KERN_WARNING "[nvdebug] Warning: WFI block at %#018llx (phys) "
			       "contains an empty context block pointer.\n", wfi_phys);
			goto out;
		}

		// All the pointers in the WFI block are virtual, so convert the CTXSW
		// block pointer to a physical address. We should always be able to find a
		// mapping for ctxsw_virt.
		if ((err = translate_page_directory(g, inst->pdb, ctxsw_virt, &ctxsw_phys, &ctxsw_phys_aperture))) {
			printk(KERN_ERR "[nvdebug] Critical: Inconsistent GPU state; context "
			       "block pointer %#018llx (virt) cannot be found in process page "
			       "tables! Translation error %d.\n", ctxsw_virt, -err);
			return ERR_PTR(-ENOTRECOVERABLE);
		}

		// Get a dereferencible pointer to the CTXSW block (via PRAMIN; invalidates `wfi`)
		if ((ctxsw_phys = addr_to_pramin_mut(g, ctxsw_phys, ctxsw_phys_aperture)) == -1)
			goto out;
		return g->regs + ctxsw_phys + NV_PRAMIN;
	} else {
		// Without subcontexts, the WFI block is the CTXSW block (ex: Pascal)
		return wfi;
	}
out:
	return NULL;
}

/* Change the preemption type to be used on a context's budget expiration
  @param chan_id As context IDs are hard to obtain and use, this function takes
                 a channel ID and looks up and modifies the associated context.
  @param rl_id   Which channel RAM address space is this channel ID in? (Not
                 used on pre-Ampere GPUs.)
  @param mode    Preemption mode to set.
  @return 0 or -errno on error

  Note: This change will not apply if the channel's context has running work,
        or if the GPU is idle and this channel's context was last to run.
        Please ensure some other task is running before calling this API.
*/
int set_channel_preemption_mode(struct nvdebug_state *g, uint32_t chan_id,
                                uint32_t rl_id,
                                enum COMPUTE_PREEMPT_TYPE mode) {
	uint64_t instance_ptr = 0;
	enum INST_TARGET instance_target;
	instance_ctrl_t *inst = NULL;
	context_switch_ctrl_t *ctxsw = NULL;
	struct runlist_iter rl_iter;
	uint32_t ctxsw_timeout_pri_base = NV_PFIFO_ENG_CTXSW_TIMEOUT;
	// Obtain the instance block
	if (g->chip_id < NV_CHIP_ID_AMPERE) {
		// Pre-Ampere, Channel RAM includes instance block pointers
		channel_ctrl_t chan;
		if (chan_id > MAX_CHID)
			return -ERANGE;
		if ((chan.raw = nvdebug_readq(g, NV_PCCSR_CHANNEL_INST(chan_id))) == -1)
			return -EIO;
		instance_ptr = (uint64_t)chan.inst_ptr << 12;
		instance_target = chan.inst_target;
	} else {
		// Starting with Ampere, instance block pointers are only included in
		// runlist entries. Something like this could work on Maxwell+, but
		// access via Channel RAM is more heavily-tested.
		struct gv100_runlist_chan* chan;
		int err;
		loff_t pos = 0;
		// Based off logic of switch_to_tsg_file_write() in runlist_procfs.c
		if ((err = get_runlist_iter(g, rl_id, &rl_iter)))
			return err;
		while (pos < rl_iter.len && !instance_ptr) {
			for_chan_in_tsg(g, chan, rl_iter.curr_entry) {
				if (chan_id == chid(g, chan)) {
					// Channel entry found in runlist. Extract instance ptr.
					instance_ptr = (uint64_t)chan->inst_ptr_hi << 32;
					instance_ptr |= (uint64_t)inst_ptr_lo(g, chan) << 12;
					instance_target = inst_target(g, chan);
					break;
				}
			}
			pos += 1 + tsg_length(g, rl_iter.curr_entry);
			rl_iter.curr_entry = next_tsg(g, rl_iter.curr_entry);
		}
		// Context switch timeout configuration register was moved with Ampere+
		ctxsw_timeout_pri_base = rl_iter.runlist_pri_base + NV_RUNLIST_ENGINE_CTXSW_TIMEOUT_CONFIG(0);
	}
	if (!instance_ptr)
		return -ENOENT;
	// Obtain an instance block pointer routed via BAR2 or SYS_MEM
	inst = instance_deref(g, instance_ptr, instance_target);
	if (IS_ERR_OR_NULL(inst))
		return PTR_ERR(inst);
	// Obtain pointer to CTXSW block routed via PRAMIN (the CTXSW block
	// does not appear to be mapped into BAR2).
	ctxsw = get_ctxsw(g, inst);
	if (IS_ERR_OR_NULL(ctxsw))
		return PTR_ERR(ctxsw);
	ctxsw->compute_preemption_options = mode;
	// If switching to a preemption mode that runs blocks or kernels non-
	// -preemptively (CTA-level and WFI respectively), disable the context switch
	// timeout. If switching to compute-instruction-level preemption (CILP),
	// reenable it. Observed to be necessary on (at least) gv11b, tu102, and ga10b
	// XXX: On ga10b (at least), the timeout configuration is reset on a resume
	//      from suspend, overwriting the change made here. This causes a CTXSW
	//      TIMEOUT interrupt to be triggered if any application tries to run
	//      non-preemptively for longer than the timeout period (3100ms on gv11b
	//      and ga10b).
	if (g->chip_id >= NV_CHIP_ID_VOLTA) {
		ctxsw_timeout_t timeout_config;
		if ((timeout_config.raw = nvdebug_readl(g, ctxsw_timeout_pri_base)) == -1)
			return -EIO;
		printk(KERN_DEBUG "[nvdebug] Previous Ctx. Sw. Timeout Configuration: period %d %s\n", timeout_config.period, timeout_config.enabled ? "enabled" : "disabled");
		timeout_config.enabled = mode == PREEMPT_CILP;
		nvdebug_writel(g, ctxsw_timeout_pri_base, timeout_config.raw);
	}
	return 0;
}
