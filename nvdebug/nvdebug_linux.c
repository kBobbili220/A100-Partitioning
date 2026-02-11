/* Copyright 2024-2025 Joshua Bakita
 * Implementation of Kernel-specific function implementations
 */
#include "nvdebug_linux.h"
#include <asm/io.h> // For read[l,q] and write[l,q]
#include <linux/pm_runtime.h> // For pm_runtime_[enabled,get,put]()

u32 nvdebug_readl(struct nvdebug_state *s, u32 r) {
	u32 ret;
	// If this is an integrated ("platform") GPU, make sure that it's on first
	// (pm_runtime_enabled() will return false until nvgpu is started. Once
	// nvgpu is started, pm_runtime_get_sync() will attempt to resume the GPU.
	// This still increments the usage counter on failure, so we undo that with
	// pm_runtime_put_noidle(). We avoid pm_runtime_resume_and_get() as it was
	// not added until Linux 5.9.11)
	// This works to bring up the TX2, Xavier, and Orin, but not the TX1.
	if (s->platd && (!pm_runtime_enabled(s->dev) || pm_runtime_get_sync(s->dev) < 0)) {
		printk(KERN_ERR "[nvdebug] nvdebug_readl: Unable to read; registers unavailable. Is GPU on?\n");
		pm_runtime_put_noidle(s->dev); // No-op if !pm_runtime_enabled()
		return -1;
	}
	ret = readl(s->regs + r);
	// If an integrated GPU, allow it to suspend again (if idle)
	if (s->platd)
		pm_runtime_put(s->dev);
	// According to open-gpu-kernel-modules, the GPU "will return 0xbad in the
	// upper 3 nibbles when there is a possible issue". Further code uses the
	// middle three nibbles as an error code, and ignores the bottom two.
	if ((ret & 0xfff00000) == 0xbad00000) {
		printk(KERN_ERR "[nvdebug] nvdebug_readl: Unable to read from register offset %#x; bad data of %#10x\n", r, ret);
		// It would be best to check INTR_0_PRI_* error is pending, to verify
		// that this was actually a bad read. Possible future work...
		// Generally a failure here in the context of nvdebug indicates that a
		// register does not exist on this platform, but one can know for sure
		// by checking which NV_PPRIV_SYS_PRI_ERROR_CODE_* define the bad read
		// matches.
		return -1;
	}
	return ret;
}

// quadword (8-byte) version of nvdebug_readl()
u64 nvdebug_readq(struct nvdebug_state *s, u32 r) {
	u64 ret;
	// If this is an integrated ("platform") GPU, make sure that it's on first
	if (s->platd && (!pm_runtime_enabled(s->dev) || pm_runtime_get_sync(s->dev) < 0)) {
		printk(KERN_ERR "[nvdebug] nvdebug_readq: Unable to read; registers unavailable. Is GPU on?\n");
		pm_runtime_put_noidle(s->dev); // No-op if !pm_runtime_enabled()
		return -1;
	}
	// readq seems to always (?) return the uppermost 32 bits as 0, so workaround with readl
	ret = readl(s->regs + r);
	ret |= ((u64)readl(s->regs + r + 4)) << 32;
	// If an integrated GPU, allow it to suspend again (if idle)
	if (s->platd)
		pm_runtime_put(s->dev);
	// See comment in nvdebug_readl() regarding error checking
	if ((ret & 0xfff00000ull) == 0xbad00000ull) {
		printk(KERN_ERR "[nvdebug] nvdebug_readq: Unable to read from register offset %#x; bad data of %#18llx\n", r, ret);
		return -1;
	}
	return ret;
}

void nvdebug_writel(struct nvdebug_state *s, u32 r, u32 v) {
	// If this is an integrated ("platform") GPU, make sure that it's on first
	if (s->platd && (!pm_runtime_enabled(s->dev) || pm_runtime_get_sync(s->dev) < 0)) {
		printk(KERN_ERR "[nvdebug] nvdebug_writel: Unable to write; registers unavailable. Is GPU on?\n");
		pm_runtime_put_noidle(s->dev); // No-op if !pm_runtime_enabled()
		return;
	}
	writel_relaxed(v, s->regs + r);
	wmb();
	// If an integrated GPU, allow it to suspend again (if idle)
	if (s->platd)
		pm_runtime_put(s->dev);
}

// quadword (8-byte) version of nvdebug_writel()
// XXX: Not clear this works on all platforms
void nvdebug_writeq(struct nvdebug_state *s, u32 r, u64 v) {
	// If this is an integrated ("platform") GPU, make sure that it's on first
	if (s->platd && (!pm_runtime_enabled(s->dev) || pm_runtime_get_sync(s->dev) < 0)) {
		printk(KERN_ERR "[nvdebug] nvdebug_writeq: Unable to write; registers unavailable. Is GPU on?\n");
		pm_runtime_put_noidle(s->dev); // No-op if !pm_runtime_enabled()
		return;
	}
	writeq_relaxed(v, s->regs + r);
	wmb();
	// If an integrated GPU, allow it to suspend again (if idle)
	if (s->platd)
		pm_runtime_put(s->dev);
}
