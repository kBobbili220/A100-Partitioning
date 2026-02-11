#include <linux/seq_file.h> // For seq_* functions and types
#include <linux/version.h>  // Macros to detect kernel version
#include <linux/platform_device.h> // For platform_get_resource()
#include <linux/pci.h> // For pci_resource_start()
#include <linux/iommu.h> // For iommu_ functions
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5,10,0)
#include <linux/dma-map-ops.h> // For get_dma_ops()
#endif

#include "nvdebug_linux.h"

// Uncomment to expand channel status, instance, and context information when
// printing the runlist
#define DETAILED_CHANNEL_INFO

#ifdef DETAILED_CHANNEL_INFO
// Print the channel instance and context swtich blocks
// XXX: THIS IS UNSAFE ON KEPLER!
// instance_deref() will call into the page table logic, which may move PRAMIN
// PRAMIN appears heavily utilized by the driver on Bonham (at least), and
// moving it causes problems.
static int runlist_detail_seq_show_inst(struct seq_file *s, struct nvdebug_state *g, char *prefix, uint64_t instance_ptr, enum INST_TARGET instance_target) {
	instance_ctrl_t *inst = NULL;
	context_switch_ctrl_t *ctxsw = NULL;
	int i;

#ifdef FALLBACK_TO_PRAMIN
	uint32_t window_reg;
	if ((g->chip_id >= NV_CHIP_ID_HOPPER && g->chip_id < NV_CHIP_ID_ADA) || g->chip_id >= NV_CHIP_ID_BLACKWELL)
		window_reg = NV_XAL_EP_BAR0_WINDOW_BASE;
	else
		window_reg = NV_PBUS_BAR0_WINDOW;
	bar0_window_t win;
	win.raw = nvdebug_readl(g, window_reg);
	inst = g->regs + NV_PRAMIN + addr_to_pramin_mut(g, instance_ptr, instance_target);
#else
	if (IS_ERR(inst = instance_deref(g, instance_ptr, instance_target)))
		return PTR_ERR(ctxsw);
#endif // FALLBACK_TO_PRAMIN
	// If unable to access instance block, skip
	if (!inst)
		return 0;

	// Print the channel instance block
	// As an ID, use upper 52 bits of the instance address (lower 12 are zero)
	//seq_printf(s, "%s+- Inst %-13llx-+\n", prefix, instance_ptr >> 12);
	seq_printf(s, "%s|= Instance Block ====|\n", prefix);
	seq_printf(s, "%s| Target Engine:    %2d|\n", prefix, inst->fc_target);
	seq_printf(s, "%s| Privileged:        %1d|\n", prefix, inst->fc_config_is_priv);
	seq_printf(s, "%s| Channel VEID:     %2d|\n", prefix, inst->fc_chan_info_veid);
	seq_printf(s, "%s| WFI PTR:            |\n", prefix);
	seq_printf(s, "%s|   %#018llx|\n", prefix, (uint64_t)inst->engine_wfi_ptr << 12);
	seq_printf(s, "%s| %20s|\n", prefix, target_to_text(inst->engine_wfi_target));
	seq_printf(s, "%s| Virtual address?   %d|\n", prefix, inst->engine_wfi_is_virtual);
	seq_printf(s, "%s| WFI VEID:         %2d|\n", prefix, inst->engine_wfi_veid);
	seq_printf(s, "%s| All PDB PTR:        |\n", prefix);
	seq_printf(s, "%s|   %#018llx|\n", prefix,  (u64)inst->pdb.page_dir << 12);
	seq_printf(s, "%s| %20s|\n", prefix, target_to_text(inst->pdb.target));
	seq_printf(s, "%s| %20s|\n", prefix, inst->pdb.is_volatile ? "volatile" : "non-volatile");
//	seq_printf(s, "%s|raw:       %0#10lx|\n", prefix, inst->pdb.raw);
	seq_printf(s, "%s| Num subcontexts:  %2ld|\n", prefix, hweight64(inst->subcontext_pdb_valid));
	// Print configuration of every enabled subcontext
	for (i = 0; i < 64; i++) {
		// Skip subcontexts without their enable bit set
		if (!(1 & (inst->subcontext_pdb_valid >> i)))
			continue;
		seq_printf(s, "%s| CPU SC%02d ASID%7d|\n", prefix, i, inst->subcontext[i].pasid);
		seq_printf(s, "%s| SC%02d PDB PTR:       |\n", prefix, i);
		seq_printf(s, "%s|   %#018llx|\n", prefix,  ((u64)inst->subcontext[i].pdb.page_dir_hi << 32) | ((u64)inst->subcontext[i].pdb.page_dir_lo << 12));
		seq_printf(s, "%s| %20s|\n", prefix, target_to_text(inst->subcontext[i].pdb.target));
		seq_printf(s, "%s| %20s|\n", prefix, inst->subcontext[i].pdb.is_volatile ? "volatile" : "non-volatile");
//		seq_printf(s, "%s|raw:       %0#10lx|\n", prefix, inst->subcontext[i].pdb.raw);
	}

	// XXX: CTXSW is only accessible via PRAMIN. Accessing PRAMIN appears to
	// either be broken, or race with the driver on Kepler (gk104 tested). So,
	// do not attempt to touch the CTXSW block on Kepler.
	// TODO: This check should be moved into addr_to_pramin_mut().
	if (g->chip_id < NV_CHIP_ID_MAXWELL)
		return 0;
	// End XXX

	if (IS_ERR(ctxsw = get_ctxsw(g, inst))) {
#ifdef FALLBACK_TO_PRAMIN
		nvdebug_writel(g, window_reg, win.raw);
#endif
		return PTR_ERR(ctxsw);
	}
	// If unable to access CTXSW block, skip
	if (!ctxsw) {
#ifdef FALLBACK_TO_PRAMIN
		nvdebug_writel(g, window_reg, win.raw);
#endif
		return 0;
	}
	// Access and print the preemption mode and context ID
	seq_printf(s, "%s|= Context State =====|\n", prefix);
	seq_printf(s, "%s| Ctx. ID:  %#10x|\n", prefix, ctxsw->context_id);
	// No other CTXSW fields are supported pre-Pascal
	if (g->chip_id < NV_CHIP_ID_PASCAL)
		return 0;
	seq_printf(s, "%s| Gfx. Preemption:%4s|\n", prefix,
	           graphics_preempt_type_to_text(ctxsw->graphics_preemption_options));
	seq_printf(s, "%s| Cmp. Preemption:%4s|\n", prefix,
	           compute_preempt_type_to_text(ctxsw->compute_preemption_options));
	seq_printf(s, "%s| #WFI Saves:%9d|\n", prefix, ctxsw->num_wfi_save_operations);
	seq_printf(s, "%s| #CTA Saves:%9d|\n", prefix, ctxsw->num_cta_save_operations);
	seq_printf(s, "%s| #GFXP Saves:%8d|\n", prefix, ctxsw->num_gfxp_save_operations);
	seq_printf(s, "%s| #CILP Saves:%8d|\n", prefix, ctxsw->num_cilp_save_operations);
#ifdef FALLBACK_TO_PRAMIN
	nvdebug_writel(g, window_reg, win.raw);
#endif
	return 0;
}

/* Print channel details using PCCSR (Programmable Channel Control System RAM?)
  @param s      Pointer to state from seq_file subsystem to pass to seq_printf
  @param g      Pointer to our internal GPU state
  @param chid   ID of channel to print details on, range [0, 512)
  @param prefix Text string to prefix each line with, or empty string
*/
static int runlist_detail_seq_show_chan(struct seq_file *s, struct nvdebug_state *g, uint32_t chid, char *prefix) {
	channel_ctrl_t chan;
	uint64_t instance_ptr;

	if ((chan.raw = nvdebug_readq(g, NV_PCCSR_CHANNEL_INST(chid))) == -1)
		return -EIO;
	instance_ptr = (uint64_t)chan.inst_ptr << 12;
	// Don't print write-only fields
	seq_printf(s, "%s|= Channel Info ======|\n", prefix);
	seq_printf(s, "%s| Enabled:           %d|\n", prefix, chan.enable);
	seq_printf(s, "%s| Next:              %d|\n", prefix, chan.next);
	seq_printf(s, "%s| PBDMA Faulted:     %d|\n", prefix, chan.pbdma_faulted);
	seq_printf(s, "%s| ENG Faulted:       %d|\n", prefix, chan.eng_faulted);
	seq_printf(s, "%s| Status:           %2d|\n", prefix, chan.status);
	seq_printf(s, "%s| Busy:              %d|\n", prefix, chan.busy);
	seq_printf(s, "%s| Instance PTR:       |\n", prefix);
	seq_printf(s, "%s|   %#018llx|\n", prefix, instance_ptr);
	seq_printf(s, "%s| %20s|\n", prefix, target_to_text(chan.inst_target));
	seq_printf(s, "%s| Instance bound:    %d|\n", prefix, chan.inst_bind);
	// Print instance block
	return runlist_detail_seq_show_inst(s, g, prefix, instance_ptr, chan.inst_target);
}

/* `runlist_detail_seq_show_chan()`, but for Ampere+
  @param instance_ptr     Address for the channel instance block
  @param instance_target  Aperture of `instance_ptr`
  @param runlist_pri_base Base of the RLRAM region for this runlist

  `runlist_pri_base` is necessary, since Channel RAM is now per-runlist on
  Ampere+, and its location is configured in Runlist RAM.
*/
static int runlist_detail_seq_show_chan_ga100(struct seq_file *s, struct nvdebug_state *g, uint32_t chid, char *prefix, uint32_t runlist_pri_base, uint64_t instance_ptr, enum INST_TARGET instance_target) {
	runlist_channel_config_t channel_config;
	channel_ctrl_ga100_t chan;

	// Channel RAM is subsidiary to Runlist RAM (ie. per-runlist) on Ampere+
	if ((channel_config.raw = nvdebug_readl(g, runlist_pri_base + NV_RUNLIST_CHANNEL_CONFIG_GA100)) == -1)
		return -EIO;
	if ((chan.raw = nvdebug_readl(g, (((uint32_t)channel_config.bar0_offset << 4) + chid * 4))) == -1)
		return -EIO;
	seq_printf(s, "%s|= Channel Info ======|\n", prefix);
	seq_printf(s, "%s| Enabled:           %d|\n", prefix, chan.enable);
	seq_printf(s, "%s| Next:              %d|\n", prefix, chan.next);
	seq_printf(s, "%s| Busy:              %d|\n", prefix, chan.busy);
	seq_printf(s, "%s| PBDMA Faulted:     %d|\n", prefix, chan.pbdma_faulted);
	seq_printf(s, "%s| ENG Faulted:       %d|\n", prefix, chan.eng_faulted);
	seq_printf(s, "%s| On PBDMA:          %d|\n", prefix, chan.on_pbdma);
	seq_printf(s, "%s| On ENG:            %d|\n", prefix, chan.on_eng);
	seq_printf(s, "%s| Pending:           %d|\n", prefix, chan.pending);
	seq_printf(s, "%s| CTX Reload:        %d|\n", prefix, chan.ctx_reload);
	seq_printf(s, "%s| PBDMA Busy:        %d|\n", prefix, chan.pbdma_busy);
	seq_printf(s, "%s| ENG Busy:          %d|\n", prefix, chan.eng_busy);
	seq_printf(s, "%s| Acquire Fail:      %d|\n", prefix, chan.acquire_fail);
	return runlist_detail_seq_show_inst(s, g, prefix, instance_ptr, instance_target);
}
#endif

#if LINUX_VERSION_CODE < KERNEL_VERSION(4,19,0)
// Bug workaround. See comment in runlist_file_seq_start()
static loff_t pos_fixup;
#endif

static void *runlist_file_seq_start(struct seq_file *s, loff_t *pos) {
	static struct runlist_iter rl_iter;
	struct nvdebug_state *g = &g_nvdebug_state[file2parentgpuidx(s->file)];
	// *pos == 0 for first call after read of file
	if (*pos == 0) {
		int err = get_runlist_iter(g, seq2gpuidx(s), &rl_iter);
		if (err)
			return ERR_PTR(err);
		// Don't try to print an empty runlist
		if (rl_iter.len <= 0)
			return NULL;
		return &rl_iter;
	}
	// If we're resuming an earlier print
	if (*pos < rl_iter.len) {
#if LINUX_VERSION_CODE < KERNEL_VERSION(4,19,0)
		// There's a nasty bug prior to 4.19-rc1 that if the buffer overflows, the
		// last update to `pos` is not saved. Work around that here by reloading a
		// saved copy of `pos`.
		if (!pos_fixup)
			return NULL;
		*pos = pos_fixup;
#endif
		return &rl_iter;
	}
	// When called with *pos != 0, we already traversed the runlist
	return NULL;
}

static void* runlist_file_seq_next(struct seq_file *s, void *raw_rl_iter,
				   loff_t *pos) {
	struct runlist_iter* rl_iter = raw_rl_iter;
	void *ret = NULL;
	struct nvdebug_state *g = &g_nvdebug_state[file2parentgpuidx(s->file)];
	// Advance by one TSG or channel
	(*pos)++;
	rl_iter->curr_entry += NV_RL_ENTRY_SIZE(g);
	// Verify we haven't reached the end of the runlist
	// len is the num of tsg entries + total num of channel entries
	if (*pos < rl_iter->len) {
		ret = rl_iter;
	}
#if LINUX_VERSION_CODE < KERNEL_VERSION(4,19,0)
	// Bug workaround. See comment in runlist_file_seq_start()
	pos_fixup = ret ? *pos : 0;
#endif
	if (rl_iter->entries_left_in_tsg)
		rl_iter->entries_left_in_tsg--;
	return ret;
}

static void runlist_file_seq_stop(struct seq_file *s, void *raw_rl_iter) {
	// No cleanup needed
}

// _show() must be idempotent. This function will be rerun if the seq_printf
// buffer was too small.
static int runlist_file_seq_show(struct seq_file *s, void *raw_rl_iter) {
	struct runlist_iter *rl_iter = raw_rl_iter;
	void *entry = rl_iter->curr_entry;
	struct nvdebug_state *g = &g_nvdebug_state[file2parentgpuidx(s->file)];
	if (entry_type(g, entry) == ENTRY_TYPE_TSG) {
		if (rl_iter->entries_left_in_tsg) {
			printk(KERN_WARNING "[nvdebug] Found TSG ID%d @ %px when %d channels were still expected under the previous TSG in the runlist!\n", tsgid(g, entry), entry, rl_iter->entries_left_in_tsg);
			while (rl_iter->entries_left_in_tsg--)
				seq_printf(s, "[missing channel]\n");
		}
		rl_iter->entries_left_in_tsg = tsg_length(g, entry) + 1;
		seq_printf(s, "+---- TSG Entry %-3d---+\n", tsgid(g, entry));
		seq_printf(s, "| Scale: %-13d|\n", timeslice_scale(g, entry));
		seq_printf(s, "| Timeout: %-11d|\n", timeslice_timeout(g, entry));
		seq_printf(s, "| Length: %-12d|\n", tsg_length(g, entry));
		seq_printf(s, "+---------------------+\n");
	} else {
		char *indt = "";
		u64 instance_ptr = 0;
		if (rl_iter->entries_left_in_tsg)
			indt = "  ";
		// Reconstruct pointer to channel instance block
		if (g->chip_id >= NV_CHIP_ID_VOLTA) {
			instance_ptr = ((struct gv100_runlist_chan*)entry)->inst_ptr_hi;
			instance_ptr <<= 32;
		}
		instance_ptr |= inst_ptr_lo(g, entry) << 12;
		// Print channel information from runlist
		seq_printf(s, "%s+- Channel Entry %-4d-+\n", indt, chid(g, entry));
		if (g->chip_id >= NV_CHIP_ID_VOLTA)
			seq_printf(s, "%s| Runqueue Selector: %d|\n", indt,
			           ((struct gv100_runlist_chan*)entry)->runqueue_selector);
		// Not populated on Kepler [ex: gk104 in Bonham (Quadro K5000)], and
		// populated but unused on Pascal [ex: gp104 in Bonham (GTX 1080 Ti)].
		// (The aperture field may be incorrectly populated as INVALID, but the
		// context still works on the aformentioned Pascal GPU.)
		seq_printf(s, "%s| Instance PTR:       |\n", indt);
		seq_printf(s, "%s|   %#018llx|\n", indt, instance_ptr);
		seq_printf(s, "%s| %20s|\n", indt, target_to_text(inst_target(g, entry)));
#ifdef DETAILED_CHANNEL_INFO
		// Print channel info from PCCSR/Channel RAM and the instance block
		if (g->chip_id < NV_CHIP_ID_AMPERE)
			runlist_detail_seq_show_chan(s, g, chid(g, entry), indt);
		else
			runlist_detail_seq_show_chan_ga100(s, g, chid(g, entry), indt, rl_iter->runlist_pri_base, instance_ptr, inst_target(g, entry));
#endif
		seq_printf(s, "%s+---------------------+\n", indt);
	}
	return 0;
}

static const struct seq_operations runlist_file_seq_ops = {
	.start = runlist_file_seq_start,
	.next = runlist_file_seq_next,
	.stop = runlist_file_seq_stop,
	.show = runlist_file_seq_show,
};

static int runlist_file_open(struct inode *inode, struct file *f) {
	return seq_open(f, &runlist_file_seq_ops);
}

struct file_operations runlist_file_ops = {
	.open = runlist_file_open,
	.read = seq_read,
	.llseek = seq_lseek,
	.release = seq_release,
};

ssize_t preempt_tsg_file_write(struct file *f, const char __user *buffer,
                               size_t count, loff_t *off) {
	uint32_t target_tsgid, target_runlist_ram;
	struct nvdebug_state *g = &g_nvdebug_state[file2parentgpuidx(f)];
	// Passing 0 as the base to kstrtou32 indicates autodetect hex/octal/dec
	int err = kstrtou32_from_user(buffer, count, 0, &target_tsgid);
	if (err)
		return err;

	// TSG IDs are a 12-bit field, so make sure the request is in-range
	if (target_tsgid > MAX_TSGID)
		return -ERANGE;

	// (Ab)use the PDE_DATA field for the index into which Runlist RAM this TSG
	// ID is scoped to (only applicable on Ampere+)
	if (g->chip_id >= NV_CHIP_ID_AMPERE)
		target_runlist_ram = file2gpuidx(f);
	else
		target_runlist_ram = 0;

	// Execute preemption
	if ((err = preempt_tsg(g, target_runlist_ram, target_tsgid)))
		return err;

	return count;
}

struct file_operations preempt_tsg_file_ops = {
	.write = preempt_tsg_file_write,
	.llseek = default_llseek,
};

ssize_t resubmit_runlist_file_write(struct file *f, const char __user *buffer,
                                    size_t count, loff_t *off) {
	uint32_t target_runlist, target_offset;
	struct nvdebug_state *g = &g_nvdebug_state[file2gpuidx(f)];
	// Passing 0 as the base to kstrtou32 indicates autodetect hex/octal/dec
	int err = kstrtou32_from_user(buffer, count, 0, &target_offset);
	if (err)
		return err;
	// (Ab)use the PDE_DATA field for the runlist ID
	target_runlist = file2gpuidx(f);

	// resubmit_runlist() checks that target_runlist is valid
	if ((err = resubmit_runlist(g, target_runlist, target_offset)))
		return err;

	return count;
}

struct file_operations resubmit_runlist_file_ops = {
	.write = resubmit_runlist_file_write,
	.llseek = default_llseek,
};


ssize_t disable_channel_file_write(struct file *f, const char __user *buffer,
                                   size_t count, loff_t *off) {
	uint32_t target_channel;
	struct nvdebug_state *g = &g_nvdebug_state[file2parentgpuidx(f)];
	// Passing 0 as the base to kstrtou32 indicates autodetect hex/octal/dec
	int err = kstrtou32_from_user(buffer, count, 0, &target_channel);
	if (err)
		return err;

	if (g->chip_id < NV_CHIP_ID_AMPERE) {
		channel_ctrl_t chan;
		if (target_channel > MAX_CHID)
			return -ERANGE;
		// Read current configuration
		if ((chan.raw = nvdebug_readq(g, NV_PCCSR_CHANNEL_INST(target_channel))) == -1)
			return -EIO;
		// Request disablement
		chan.enable_clear = true;
		nvdebug_writeq(g, NV_PCCSR_CHANNEL_INST(target_channel), chan.raw);
	} else {
		uint32_t runlist_reg_base, chram_base, channel_max;
		runlist_channel_config_t channel_config;
		channel_ctrl_ga100_t chan;
		// (Ab)use the PDE_DATA field for the runlist ID
		if ((err = get_runlist_ram(g, file2gpuidx(f), &runlist_reg_base)))
			return err;
		// Channel RAM is subsidiary to Runlist RAM (ie. per-runlist) on Ampere
		if ((channel_config.raw = nvdebug_readl(g, runlist_reg_base + NV_RUNLIST_CHANNEL_CONFIG_GA100)) == -1)
			return -EIO;
		channel_max = 1u << channel_config.num_channels_log2;
		if (target_channel >= channel_max)
			return -ERANGE;
		chram_base = (uint32_t)channel_config.bar0_offset << 4;
		// Writing zeros to any field of the Ampere+ channel control structure
		// does nothing, so don't bother to read the structure first, and just
		// write zeros to all the fields we don't care about.
		chan.raw = 0;
		chan.is_write_one_clears_bits = 1; // Invert meaning of writing 1
		chan.enable = 1;
		nvdebug_writel(g, chram_base + sizeof(channel_ctrl_ga100_t) * target_channel, chan.raw);
	}

	return count;
}

struct file_operations disable_channel_file_ops = {
	.write = disable_channel_file_write,
	.llseek = default_llseek,
};

ssize_t enable_channel_file_write(struct file *f, const char __user *buffer,
                                  size_t count, loff_t *off) {
	uint32_t target_channel;
	struct nvdebug_state *g = &g_nvdebug_state[file2parentgpuidx(f)];
	// Passing 0 as the base to kstrtou32 indicates autodetect hex/octal/dec
	int err = kstrtou32_from_user(buffer, count, 0, &target_channel);
	if (err)
		return err;

	if (g->chip_id < NV_CHIP_ID_AMPERE) {
		channel_ctrl_t chan;
		if (target_channel > MAX_CHID)
			return -ERANGE;
		// Read current configuration
		if ((chan.raw = nvdebug_readq(g, NV_PCCSR_CHANNEL_INST(target_channel))) == -1)
			return -EIO;
		// Disable channel
		chan.enable_set = true;
		nvdebug_writeq(g, NV_PCCSR_CHANNEL_INST(target_channel), chan.raw);
	} else {
		uint32_t runlist_reg_base, chram_base, channel_max;
		runlist_channel_config_t channel_config;
		channel_ctrl_ga100_t chan;
		// (Ab)use the PDE_DATA field for the runlist ID
		if ((err = get_runlist_ram(g, file2gpuidx(f), &runlist_reg_base)))
			return err;
		// Channel RAM is subsidiary to Runlist RAM (ie. per-runlist) on Ampere
		if ((channel_config.raw = nvdebug_readl(g, runlist_reg_base + NV_RUNLIST_CHANNEL_CONFIG_GA100)) == -1)
			return -EIO;
		channel_max = 1u << channel_config.num_channels_log2;
		if (target_channel >= channel_max)
			return -ERANGE;
		chram_base = (uint32_t)channel_config.bar0_offset << 4;
		// Writing zeros to any field of the Ampere+ channel control structure
		// does nothing, so don't bother to read the structure first, and just
		// write zeros to all the fields we don't care about.
		chan.raw = 0;
		chan.enable = 1;
		nvdebug_writel(g, chram_base + sizeof(channel_ctrl_ga100_t) * target_channel, chan.raw);
	}

	return count;
}

struct file_operations enable_channel_file_ops = {
	.write = enable_channel_file_write,
	.llseek = default_llseek,
};

ssize_t comm_preempt_channel_file_write(struct file *f, const char __user *buf,
                                        size_t count, loff_t *off,
                                        enum COMPUTE_PREEMPT_TYPE mode) {
	uint32_t target_channel, target_runlist;
	struct nvdebug_state *g = &g_nvdebug_state[file2parentgpuidx(f)];
	// Passing 0 as the base to kstrtou32 indicates autodetect hex/octal/dec
	int err = kstrtou32_from_user(buf, count, 0, &target_channel);
	if (err)
		return err;
	// (Ab)use the PDE_DATA field used by file2gpuidx() for the runlist ID
	target_runlist = file2gpuidx(f);
	// Set preemption mode for the context of this channel
	if ((err = set_channel_preemption_mode(g, target_channel, target_runlist, mode)))
		return err;

	return count;
}

ssize_t wfi_preempt_channel_file_write(struct file *f, const char __user *buf,
                                       size_t count, loff_t *off) {
	return comm_preempt_channel_file_write(f, buf, count, off, PREEMPT_WFI);
}

struct file_operations wfi_preempt_channel_file_ops = {
	.write = wfi_preempt_channel_file_write,
	.llseek = default_llseek,
};

ssize_t cta_preempt_channel_file_write(struct file *f, const char __user *buf,
                                       size_t count, loff_t *off) {
	return comm_preempt_channel_file_write(f, buf, count, off, PREEMPT_CTA);
}

struct file_operations cta_preempt_channel_file_ops = {
	.write = cta_preempt_channel_file_write,
	.llseek = default_llseek,
};

ssize_t cil_preempt_channel_file_write(struct file *f, const char __user *buf,
                                       size_t count, loff_t *off) {
	return comm_preempt_channel_file_write(f, buf, count, off, PREEMPT_CILP);
}

struct file_operations cil_preempt_channel_file_ops = {
	.write = cil_preempt_channel_file_write,
	.llseek = default_llseek,
};

// Tested working on Pascal (gp106) through Ada (ad102)
ssize_t switch_to_tsg_file_write(struct file *f, const char __user *buffer,
                                 size_t count, loff_t *off) {
	uint32_t target_tsgid, target_runlist, channel_regs_base;
	struct gv100_runlist_chan* chan;
	channel_ctrl_t chan_ctl;
	channel_ctrl_ga100_t chan_ctl_ga100;
	struct runlist_iter rl_iter;
	loff_t pos = 0;
	struct nvdebug_state *g = &g_nvdebug_state[file2parentgpuidx(f)];
	// Passing 0 as the base to kstrtou32 indicates autodetect hex/octal/dec
	int err = kstrtou32_from_user(buffer, count, 0, &target_tsgid);
	if (err)
		return err;

	if (target_tsgid > MAX_TSGID)
		return -ERANGE;

	// (Ab)use the PDE_DATA field for the runlist ID
	target_runlist = file2gpuidx(f);

	if ((err = get_runlist_iter(g, target_runlist, &rl_iter)))
		return err;

	// On Ampere, TSG and Channel IDs are only unique per-runlist, so we need
	// to pull the per-runlist copy of Channel RAM.
	if (g->chip_id >= NV_CHIP_ID_AMPERE) {
		uint32_t runlist_regs_base;
		runlist_channel_config_t chan_config;
		if ((err = get_runlist_ram(g, target_runlist, &runlist_regs_base)))
			return err;
		// Channel RAM is subsidiary to Runlist RAM (ie. per-runlist) on Ampere
		if ((chan_config.raw = nvdebug_readl(g, runlist_regs_base + NV_RUNLIST_CHANNEL_CONFIG_GA100)) == -1)
			return -EIO;
		channel_regs_base = (uint32_t)chan_config.bar0_offset << 4;
	}

	// Iterate through all TSGs
	while (pos < rl_iter.len) {
		bool enable = false;
		if (tsgid(g, rl_iter.curr_entry) == target_tsgid)
			enable = true;

		// Either enable or disable all channels of each TSG, dependent on if
		// they are contained within the target TSG or not.
		for_chan_in_tsg(g, chan, rl_iter.curr_entry) {
			if (g->chip_id < NV_CHIP_ID_AMPERE) {
				// Read, update, write for PCCSR
				if ((chan_ctl.raw = nvdebug_readq(g, NV_PCCSR_CHANNEL_INST(chid(g, chan)))) == -1)
					return -EIO;
				if (enable)
					chan_ctl.enable_set = true;
				else
					chan_ctl.enable_clear = true;
				nvdebug_writeq(g, NV_PCCSR_CHANNEL_INST(chid(g, chan)), chan_ctl.raw);
			} else {
				// Writing a 0 does nothing on Ampere+, so we can just write
				chan_ctl_ga100.raw = 0;
				chan_ctl_ga100.is_write_one_clears_bits = !enable;
				chan_ctl_ga100.enable = true;
				nvdebug_writel(g, channel_regs_base + sizeof(chan_ctl_ga100) * chid(g, chan), chan_ctl_ga100.raw);
			}
		}
		pos += 1 + tsg_length(g, rl_iter.curr_entry);
		rl_iter.curr_entry = next_tsg(g, rl_iter.curr_entry);

		// TODO: Fix the above for bare channels. Add "for_chan_until_tsg"?
	}
#warning switch_to_tsg has preempt_runlist omitted!
	return count;

	// Resubmit the runlist to ensure that changes to channel enablement are
	// picked up on Turing+ GPUs (channel enablements may not be otherwise).
	if (g->chip_id >= NV_CHIP_ID_TURING)
		if ((err = resubmit_runlist(g, target_runlist, -1)))
			return err;

	// Trigger a runlist-level preempt to stop whatever was running, triggering
	// the runlist scheduler to select and run the next-enabled channel.
	if ((err = preempt_runlist(g, target_runlist)))
		return err;

	return count;
}

struct file_operations switch_to_tsg_file_ops = {
	.write = switch_to_tsg_file_write,
	.llseek = default_llseek,
};

ssize_t preempt_runlist_file_write(struct file *f, const char __user *buffer,
                                    size_t count, loff_t *off) {
	uint32_t target_runlist;
	struct nvdebug_state *g = &g_nvdebug_state[file2gpuidx(f)];
	// Passing 0 as the base to kstrtou32 indicates autodetect hex/octal/dec
	int err = kstrtou32_from_user(buffer, count, 0, &target_runlist);
	if (err)
		return err;

	// TODO: Check runlist is in-range
	if ((err = preempt_runlist(g, target_runlist)))
		return err;

	return count;
}

struct file_operations preempt_runlist_file_ops = {
	.write = preempt_runlist_file_write,
	.llseek = default_llseek,
};

// Value written to this file is which runlist to ack the IRQ for
ssize_t ack_bad_tsg_file_write(struct file *f, const char __user *buffer,
                                    size_t count, loff_t *off) {
	uint32_t target_runlist;
	uint32_t rl_ram_off;
	struct nvdebug_state *g = &g_nvdebug_state[file2gpuidx(f)];
	// Passing 0 as the base to kstrtou32 indicates autodetect hex/octal/dec
	int err = kstrtou32_from_user(buffer, count, 0, &target_runlist);
	if (err)
		return err;

	if ((err = get_runlist_ram(g, target_runlist, &rl_ram_off)))
		return err;

	nvdebug_writel(g, rl_ram_off + 0x100, 1 << 12);

	return count;
}

struct file_operations ack_bad_tsg_file_ops = {
	.write = ack_bad_tsg_file_write,
	.llseek = default_llseek,
};

// Rather than mapping all of BAR0, we just map:
// - On Pascal, Volta, Turing: MC_BOOT, PFIFO, PCCSR, PTOP
// - On Ampere: MC_BOOT, RAMRL(0), CHRAM(0), PTOP
// "All CUDA-managed pointers are within---the first 40 bits of the process's
// VA space" (Sec. 4.1, GPUDirect RDMA Documentation)
// - This means 0x00ff_ffff_ffff is the highest valid CUDA virtual address,
//   and all higher addresses are unused.
// - So we use 0x6000_0000_0000+; this falls within the first PDE3 entry, and
//   at the end of the PDE2 entries
//   + Using the second PDE3 entry did not appear to work on Jetson (IIRC)
#define BAR0_USER_ADDR 0x0000700000000000llu
#define MEM_USER_ADDR 0x0000600000000000llu

/* Map all of GPU VRAM, and selected BAR0 regions, into a channel instance's
 * virtual address space at predefined offsets (above).
 *
 * @param g        Pointer to the nvdebug state for the selected GPU
 * @param inst_ptr Dereferencible pointer to the channel's instance block
 * @returns 0 on success, -errno on error
 *
 * Support: Pascal, Volta, Turing, Ampere
 */
int map_mem_for_instance(struct nvdebug_state *g, instance_ctrl_t *inst_ptr) {
	int ret;
	uintptr_t off, ram_size;
	dma_addr_t bus_mc_boot_ram, bus_ptop_ram, bus_fifo_ram, bus_chan_ctrl_ram;
	uint64_t mc_boot_ram, ptop_ram, fifo_ram, chan_ctrl_ram;
	page_dir_config_t chan_pd_config;
	memory_range_t mem_range;
	uint32_t channel_ram_off, runlist_ram_off, channel_ram_size, bar0_base;
	struct iommu_domain *dom;

	if (g->chip_id >= NV_CHIP_ID_AMPERE) {
		runlist_channel_config_t channel_config;
		if ((ret = get_runlist_ram(g, 0, &runlist_ram_off))) {
			printk(KERN_ERR "[nvdebug] %s: Unable to determine location of runlist0 RAM!\n", __func__);
			return ret;
		}
		if (runlist_ram_off & 0xfff) {
			printk(KERN_ERR "[nvdebug] %s: Runlist0 RAM is not page-aligned!\n", __func__);
			return -EAFNOSUPPORT;
		}
		if ((channel_config.raw = nvdebug_readl(g, runlist_ram_off + NV_RUNLIST_CHANNEL_CONFIG_GA100)) == -1)
			return -EIO;
		channel_ram_off = (uint32_t)channel_config.bar0_offset << 4;
		if (channel_ram_off & 0xfff) {
			printk(KERN_ERR "[nvdebug] %s: Runlist0 CHRAM is not page-aligned!\n", __func__);
			return -EAFNOSUPPORT;
		}
		channel_ram_size = (1 << channel_config.num_channels_log2) * sizeof(channel_ctrl_ga100_t);
		printk(KERN_DEBUG "[nvdebug] %s: Mapping CHRAM at %#018llx--%x and RLRAM at %#018llx--%x.\n", __func__, BAR0_USER_ADDR + channel_ram_off, channel_ram_size-1, BAR0_USER_ADDR + runlist_ram_off, 4095);
	} else {
		channel_ram_off = NV_PCCSR;
		// MAX_CHID * sizeof(channel_ctrl_gf100_t) is < 4 KiB, so hardcode
		channel_ram_size = 4096;
		runlist_ram_off = NV_PFIFO;
	}

	// map_mem_by_chid() pulls the instance block via PRAMIN, so inst_ptr will
	// be invalid after moving PRAMIN (eg. as part of a page table operation).
	// To avoid accessing inst_ptr after invalidation, keep a copy of what we
	// need.
	chan_pd_config = inst_ptr->pdb;

	// map_page_directory_v1() is unimplemented, precluding Maxwell (or older)
	// support (as they don't support v2 page tables).
	if (!chan_pd_config.is_ver2)
		return -EOPNOTSUPP;

	// Determine the size of GPU physical memory (VRAM).
	if ((mem_range.raw = nvdebug_readl(g, NV_FB_MMU_LOCAL_MEMORY_RANGE)) == -1)
		return -EIO;
	ram_size = memory_range_to_bytes(mem_range);

	// We map memory using huge pages, and thus do not support GPUs with
	// non-2-MiB-divisible VID_MEM sizes.
	if (ram_size % (1 << 21) != 0) {
		printk(KERN_ERR "[nvdebug] %s: GPU VID_MEM of %lu bytes is not a multiple of 2 MiB!\n", __func__, ram_size);
		return -EAFNOSUPPORT;
	}

	// Map all of physical GPU memory (VID_MEM) into this channels's GPU virtual
	// address space using huge (2 MiB) pages.
	for (off = 0; off < ram_size; off += (1 << 21)) {
		if ((ret = map_page_directory(g, chan_pd_config,
					MEM_USER_ADDR + off, off, TARGET_VID_MEM, true)) < 0)
			return ret;
		// If the mapping already exists for this page directory, the other
		// mappings should already exist, and can be skipped.
		if (ret == 1) {
			printk(KERN_INFO "[nvdebug] %s: VRAM mapping from %llx to %lx already exists. Assuming all mappings already exist and returning early...\n", __func__, MEM_USER_ADDR + off, off);
			return 0;
		}
	}

	// Map Channel RAM to a GPU-accessible bus address (gets past any IOMMU or
	// IOVA layers), then map that address into this channel's GPU virtual
	// address space. NV_PCCSR_CHANNEL_INST(0) is 4k-aligned, so it can be
	// directly mapped.
	// XXX: All these mappings are currently returning -1 on all reads on
	//      sunlight, jbakita-old, jetson-xavier, jetson-orin, and bonham,
	//      which seems to be returned from the PCIe root (on PCIe GPUs).
	if (g->pcid)
		bar0_base = pci_resource_start(g->pcid, 0);
	else if (g->platd)
		bar0_base = platform_get_resource(g->platd, IORESOURCE_MEM, 0)->start;
	else
		return -ENOTRECOVERABLE;
	mc_boot_ram = NV_MC_BOOT_0 + bar0_base;
	// PTOP fits within a page, but not page-aligned; round down.
	ptop_ram = (NV_PTOP & ~0xfffu) + bar0_base;
	fifo_ram = runlist_ram_off + bar0_base;
	chan_ctrl_ram = channel_ram_off + bar0_base;

	// Check if GPU-accessible bus addresses are the same as CPU-visible physical
	// addresses. Logic from amdgpu_device_check_iommu_direct_map().
	dom = iommu_get_domain_for_dev(g->dev);
	if (!dom || dom->type == IOMMU_DOMAIN_IDENTITY) {
		// Used for: jbakita-old, sunlight, jetson-xavier, jetson-orin integrated, bonham, ?
		// (For all these, reads on the mapping return only -1.)
		// (Forcing these through dma_map_resource()/iommu_map() changes nothing)
		// (Note that the `ls -l /sys/class/iommu/*/devices` also reports that the
		// GPU is not available under the I/O MMU on these platforms.)
		// To fix this, please enable AMD-Vi/ARM SMMU/Intel VT-d in your BIOS
		// settings, UEFI settings, or device-tree file. Supported on:
		// - AMD: Bulldozer+ (or Phenom II w/ 890FX or 990FX Chipset)
		// - Intel: Most since Core2 Duo
		// Note that while the Jetson Orin has an SMMU (I/O MMU), the GPU does not
		// appear to be configured by any pre-provided device tree files to use the
		// SMMU.
		printk(KERN_INFO "[nvdebug] map_mem_ctxid: I/O MMU is unavailable/disabled for GPU %x. Assuming phys and bus addresses are identical...\n", g->chip_id);
		bus_mc_boot_ram = mc_boot_ram;
		bus_ptop_ram = ptop_ram;
		bus_fifo_ram = fifo_ram;
		bus_chan_ctrl_ram = chan_ctrl_ram;
	} else {
		printk(KERN_INFO "[nvdebug] map_mem_ctxid: I/O MMU is enabled. Attempting to use dma_map_resource()...\n");
		// Used for: tama, yamaha
		// Fails on tama, yamaha
		// (Works on jetson-xavier, jetson-orin and bonham, but appears to be a no-op, and
		// yields inaccessible memory. Get `mc-err: (255) csr_nvl7r: EMEM address decode error`
		// on access on jetson boards, and a -1 read on all.)
		bus_mc_boot_ram = dma_map_resource(g->dev, mc_boot_ram, 4096*2 /* *2 is a XXX hack to include PBUS */, DMA_BIDIRECTIONAL, DMA_ATTR_SKIP_CPU_SYNC);
		bus_ptop_ram = dma_map_resource(g->dev, ptop_ram, 4096, DMA_BIDIRECTIONAL, DMA_ATTR_SKIP_CPU_SYNC);
		bus_fifo_ram = dma_map_resource(g->dev, fifo_ram, 4096*8 /* *8 is a XXX hack */, DMA_BIDIRECTIONAL, DMA_ATTR_SKIP_CPU_SYNC);
		bus_chan_ctrl_ram = dma_map_resource(g->dev, chan_ctrl_ram, 2*4096, DMA_BIDIRECTIONAL, DMA_ATTR_SKIP_CPU_SYNC);
		if (dma_mapping_error(g->dev, bus_mc_boot_ram) ||
			dma_mapping_error(g->dev, bus_ptop_ram) ||
			dma_mapping_error(g->dev, bus_fifo_ram) ||
			dma_mapping_error(g->dev, bus_chan_ctrl_ram)) {
			// Used for: tama, yamaha
			printk(KERN_WARNING "[nvdebug] map_mem_ctxid: Unable to map BAR0 addresses to device-accessible addresses via dma_map_resource(). Return codes: %d for MC_BOOT, %d for PFIFO, %d for PCCSR.\n",
				   dma_mapping_error(g->dev, bus_mc_boot_ram),
				   dma_mapping_error(g->dev, bus_fifo_ram),
				   dma_mapping_error(g->dev, bus_chan_ctrl_ram));
			// This fallback does not appear to work on jbakita-old (5.4, GART IOMMU), but works on tama
			if (!get_dma_ops(g->dev))
				printk(KERN_WARNING "[nvdebug] Reason: No DMA `ops`, and direct mapping failed.\n");
			else if (!get_dma_ops(g->dev)->map_resource)
				// Fires on: tama, yamaha
				printk(KERN_WARNING "[nvdebug] Reason: `map_resource` function undefined on this platform.\n");
			if (!dom) {
				printk(KERN_ERR "[nvdebug] map_mem_ctxid: No I/O MMU available and dma_map_resource() failed. Aborting mapping of BAR0 regions!\n");
				return -ENOTRECOVERABLE;
			}
			printk(KERN_INFO "[nvdebug] map_mem_ctxid: Trying to fall back to direct I/O MMU manipulation...\n");
			// XXX: Fallback to directly creating the I/O MMU mappings.
			//      This is necessary. Directly accessing BAR0 addresses throws I/O MMU
			//      errors in the kernel log on yamaha.
			// See also: comment on kfd_mem_dmamap_sg_bo() in amdgpu
			// Note: dma_map_resource -> map_resource -> [arm_]iommu_map_resource
			//       -> __iommu_dma_map -> iommu_map is the happy-path, but this seems to
			//       regularly fail, even though the iommu_map path works. One key
			//       difference is that the dma_map_resource() path also includes
			//       IOMMU_MMIO in the iommu_map() flags.
			bus_mc_boot_ram = mc_boot_ram;
			bus_ptop_ram = ptop_ram;
			bus_fifo_ram = fifo_ram;
			bus_chan_ctrl_ram = chan_ctrl_ram;
			// Create identity mapping
			ret = iommu_map(dom, mc_boot_ram, mc_boot_ram, 4096*2 /* *2 is a hack to fit in PBUS*/, IOMMU_READ | IOMMU_WRITE);
			if (ret < 0) {
				printk(KERN_ERR "[nvdebug] map_mem_ctxid: Attempt to bypass and go directly to I/O MMU failed for MC_BOOT!\n");
				return ret;
			}
			ret = iommu_map(dom, ptop_ram, ptop_ram, 4096, IOMMU_READ | IOMMU_WRITE);
			if (ret < 0) {
				printk(KERN_ERR "[nvdebug] map_mem_ctxid: Attempt to bypass and go directly to I/O MMU failed for PTOP!\n");
				return ret;
			}
			ret = iommu_map(dom, fifo_ram, fifo_ram, 4096*8 /* *8 is XXX hack*/, IOMMU_READ | IOMMU_WRITE);
			if (ret < 0) {
				printk(KERN_ERR "[nvdebug] map_mem_ctxid: Attempt to bypass and go directly to I/O MMU failed for FIFO!\n");
				return ret;
			}
			ret = iommu_map(dom, chan_ctrl_ram, chan_ctrl_ram, channel_ram_size, IOMMU_READ | IOMMU_WRITE);
			if (ret < 0) {
				printk(KERN_ERR "[nvdebug] map_mem_ctxid: Attempt to bypass and go directly to I/O MMU failed for PCCSR!\n");
				return ret;
			}
		}
	}
	// TARGET_SYS_MEM_NONCOHERENT tells the GPU to bypass the CPU L2 cache for
	// accesses to this memory.
	// "Clients should normally use [SYS_MEM_NON_COHERENT]" (nvgpu)
	//
	// "Non-coherent system memory.
	//  (GPU) MMU will NOT maintain coherence with CPU L2 cache.
	//  Higher-level APIs should only allow this when it is known
	//  the memory is not cacheable by CPU or the coherency is
	//  managed explicitly (e.g. w/ flushes in SW).
	//  Also consider that this path is not necessarily faster." (open-gpu-kernel-modules)
	//
	// "Coherent system memory.
	//  (GPU) MMU will snoop CPU L2 cache if possible.
	//  This is usually the safer choice over NONCOH since it works
	//  whether the memory is cached by CPU L2 or not.
	//  On some CPU architectures going through CPU L2 may
	//  even be faster than the non-coherent path." (open-gpu-kernel-modules)
	//
	// I suspect that that for SYS_MEM_NONCOHERENT mappings, the "no snoop"
	// attribute bit will be set on associated PCIe read/write transactions.
	//
	// The only other bits in a PCIe read/write transaction that could be
	// relevant are the two AT (Address Translation) bits added in PCIe 2.0.
	if ((ret = map_page_directory(g, chan_pd_config, BAR0_USER_ADDR + NV_MC_BOOT_0,
	             bus_mc_boot_ram, TARGET_SYS_MEM_NONCOHERENT, false)) < 0)
		return ret;
	// XXX
	if ((ret = map_page_directory(g, chan_pd_config, BAR0_USER_ADDR + NV_MC_BOOT_0 + 4096,
	             bus_mc_boot_ram + 4096, TARGET_SYS_MEM_NONCOHERENT, false)) < 0)
		return ret;
	if ((ret = map_page_directory(g, chan_pd_config, BAR0_USER_ADDR + (NV_PTOP & ~0xfffu),
	             bus_ptop_ram, TARGET_SYS_MEM_NONCOHERENT, false)) < 0)
		return ret;
	if ((ret = map_page_directory(g, chan_pd_config, BAR0_USER_ADDR + runlist_ram_off,
	             bus_fifo_ram, TARGET_SYS_MEM_NONCOHERENT, false)) < 0)
		return ret;
	// XXX
	for (off = 4096; off < 8*4096; off += 4096)
		if ((ret = map_page_directory(g, chan_pd_config, BAR0_USER_ADDR + runlist_ram_off+off,
					 bus_fifo_ram+off, TARGET_SYS_MEM_NONCOHERENT, false)) < 0)
			return ret;
	// Channel control RAM can span two or more pages on Ampere+
	for (off = 0; off < channel_ram_size; off += 4096)
		if ((ret = map_page_directory(g, chan_pd_config, BAR0_USER_ADDR + channel_ram_off + off,
		             bus_chan_ctrl_ram + off, TARGET_SYS_MEM_NONCOHERENT, false)) < 0)
			return ret;
	return 0;
}

// Map by context ID
// See constituent functions for info on what they do; comments not repeated.
// Tested on Pascal, Volta, Turing, and Kepler
ssize_t map_mem_ctxid_file_write(struct file *f, const char __user *buffer,
				   size_t count, loff_t *off) {
	int err, target_context, target_runlist;
	loff_t pos;
	uint64_t instance_ptr;
	enum INST_TARGET instance_target;
	struct runlist_iter rl_iter;
	instance_ctrl_t *inst;
	context_switch_ctrl_t *ctx_block;
	struct nvdebug_state *g = &g_nvdebug_state[file2parentgpuidx(f)];
	// Passing 0 as the base to kstrtou32 indicates autodetect hex/octal/dec
	if ((err = kstrtou32_from_user(buffer, count, 0, &target_context)))
		return err;
	target_runlist = file2gpuidx(f);

	// Get dereferencable pointer to the runlist
	if ((err = get_runlist_iter(g, target_runlist, &rl_iter)))
		return err;
	// Find a channel in the runlist matching the provided context ID
	for (pos = 0; pos < rl_iter.len; pos++, rl_iter.curr_entry += NV_RL_ENTRY_SIZE(g)) {
		uint32_t ctxsw_timeout_pri_base = NV_PFIFO_ENG_CTXSW_TIMEOUT;
		if (entry_type(g, rl_iter.curr_entry) == ENTRY_TYPE_TSG)
			continue;
		// Get instance block address
		if (g->chip_id >= NV_CHIP_ID_AMPERE) {
			instance_ptr = ((struct gv100_runlist_chan*)rl_iter.curr_entry)->inst_ptr_hi;
			instance_ptr <<= 32;
			instance_ptr |= (uint64_t)inst_ptr_lo(g, rl_iter.curr_entry) << 12;
			instance_target = inst_target(g, rl_iter.curr_entry);
			ctxsw_timeout_pri_base = rl_iter.runlist_pri_base + NV_RUNLIST_ENGINE_CTXSW_TIMEOUT_CONFIG(0);
		} else {
			channel_ctrl_t chan;
			chan.raw = nvdebug_readq(g, NV_PCCSR_CHANNEL_INST(chid(g, rl_iter.curr_entry)));
			if (chan.raw == -1)
				return -EIO;
			instance_ptr = (uint64_t)chan.inst_ptr << 12;
			instance_target = chan.inst_target;
		}
		// Skip channels with unconfigured or INVALID instance blocks
		if (!instance_ptr || instance_target == 1) {
			printk(KERN_WARNING "[nvdebug] Channel %d is in runlist %d, but "
			       "lacks a valid instance block", chid(g, rl_iter.curr_entry),
			       target_runlist);
			continue;
		}

		// Get a dereferencable pointer to the instance block
		if (IS_ERR(inst = instance_deref(g, instance_ptr, instance_target)))
			return PTR_ERR(inst);
		// If unable to access instance block, skip
		if (!inst)
			continue;

		// Get dereferencable pointer to CTXSW block
		if (IS_ERR(ctx_block = get_ctxsw(g, inst)))
			return PTR_ERR(ctx_block);
		// If unable to access CTXSW block, skip
		if (!ctx_block)
			continue;
		// Check if the context ID matches
		if (ctx_block->context_id != target_context)
			continue;

		// XXX: Disable the context switch timeout while we're here
		ctxsw_timeout_t timeout_config;
		if ((timeout_config.raw = nvdebug_readl(g, ctxsw_timeout_pri_base)) == -1)
			return -EIO;
		timeout_config.enabled = 0;
		nvdebug_writel(g, ctxsw_timeout_pri_base, timeout_config.raw);
		// XXX: Attempt setting preemption mode while we're here
		ctx_block->compute_preemption_options = PREEMPT_CTA;

		// Map memory and return
		if ((err = map_mem_for_instance(g, inst)) < 0)
			return err;
		return count;
	}
	return -ESRCH;
}

struct file_operations map_mem_ctxid_file_ops = {
	.write = map_mem_ctxid_file_write,
	.llseek = default_llseek,
};

// Map by channel ID (LEGACY; unclear if this needs to be kept)
// Support: Pascal, Volta, and Turing only
ssize_t map_mem_chid_file_write(struct file *f, const char __user *buffer,
				   size_t count, loff_t *off) {
	int ret, target_channel;
	struct nvdebug_state *g = &g_nvdebug_state[file2parentgpuidx(f)];
	channel_ctrl_t chan;
	instance_ctrl_t *inst_ptr;
	bool all = false;
	uint64_t inst_ptr_off;
	page_dir_config_t bar2_pd_config;
	// Passing 0 as the base to kstrtou32 indicates autodetect hex/octal/dec
	if ((ret = kstrtos32_from_user(buffer, count, 0, &target_channel)))
		return ret;

	if (g->chip_id >= NV_CHIP_ID_AMPERE)
		return -ENOSYS;

	// This API is for nvsched, which is only supported on GPUs which support
	// instruction-level preemption (Pascal+).
	if (g->chip_id < NV_CHIP_ID_PASCAL)
		return -EOPNOTSUPP;

	if (target_channel > MAX_CHID)
		return -ERANGE;

	// Passing -1 indicates that all channels should be mapped
	if (target_channel == -1) {
		all = true;
		target_channel = 0;
	}

	do {
		printk(KERN_INFO "[nvdebug] Mapping channel %d\n", target_channel);
		// Read the channel's configuration block, which includes the address of
		// this channel's instance block, which contains a page table pointer.
		// TODO: Verify this works with the channel RAM changes on Ampere+
		chan.raw = nvdebug_readq(g, NV_PCCSR_CHANNEL_INST(target_channel));
		if (chan.raw == -1)
			return -EIO;

		// If the instance pointer is unconfigured or the target is 1 (INVALID),
		// this channel is not in-use on any runlist and can be skipped.
		if (chan.inst_ptr == 0 || chan.inst_target == 1)
			continue;

		// Find page tables which define how BAR2 offsets are tranlated to physical
		// VID_MEM/SYS_MEM addresses. (We have to do this every time since we reset
		// PRAMIN.)
		if ((ret = get_bar2_pdb(g, &bar2_pd_config)) < 0)
			return ret;

		// Pascal+ GPUs use Version 2 page tables, so this shouldn't be a problem
		if (!bar2_pd_config.is_ver2)
			return -ENOSYS;

		// To read the instance block, first find where it is mapped in BAR2
		if ((inst_ptr_off = search_page_directory(g, bar2_pd_config, (u64)chan.inst_ptr << 12, chan.inst_target)) == 0) {
			// If no mapping can be found in BAR2, fallback to accessing the
			// instance block via the PRAMIN window.
			printk(KERN_WARNING "[nvdebug] Warning: Channel %d has no instance "
				   "block mapped in BAR2. Falling back to PRAMIN...\n", target_channel);
			if ((ret = addr_to_pramin_mut(g, (u64)chan.inst_ptr << 12, chan.inst_target)) < 0)
				return -EOPNOTSUPP;
			inst_ptr = g->regs + NV_PRAMIN + ret;
		} else {
			inst_ptr = g->bar2 + inst_ptr_off;
		}

		if ((ret = map_mem_for_instance(g, inst_ptr)))
			return ret;

	// If mapping all channels, start again at the next one
	} while (all && ++target_channel <= MAX_CHID);

	return count;
}

struct file_operations map_mem_chid_file_ops = {
	.write = map_mem_chid_file_write,
	.llseek = default_llseek,
};
