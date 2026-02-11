/* Copyright 2024 Joshua Bakita
 * SPDX-License-Identifier: MIT
 *
 * Helpers which are kernel-specific
 */
#include "nvdebug.h"

#include <linux/device.h>   // For dev_get_drvdata()
#include <linux/seq_file.h> // For struct seq_file
#include <linux/proc_fs.h>  // For pde_data() macro
#include <linux/version.h>  // For KERNEL_VERSION and LINUX_VERSION_CODE

static inline struct gk20a *get_gk20a(struct device *dev) {
	// Only works because gk20a* is the first member of gk20a_platform
	return *((struct gk20a**)dev_get_drvdata(dev));
}

// PDE_DATA was Renamed to pde_data in Linux 5.17 to deconflict with a driver
#if LINUX_VERSION_CODE < KERNEL_VERSION(5,17,0)
#define pde_data PDE_DATA
#endif

// iommu_map() requires an extra parameter on Linux 6.3+
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6,3,0)
#define iommu_map(a, b, c, d, e) iommu_map(a, b, c, d, e, GFP_KERNEL)
#endif

// We us the data field of the proc_dir_entry ("PDE" in this function) to store
// our index into the g_nvdebug_state array
static inline int seq2gpuidx(struct seq_file *s) {
	const struct file *f = s->file;
	return (uintptr_t)pde_data(file_inode(f));
}
static inline int file2gpuidx(const struct file *f) {
	return (uintptr_t)pde_data(file_inode(f));
}
static inline int file2parentgpuidx(const struct file *f) {
	// Should be safe to call on ProcFS entries, as our parent should (?)
	// still exist if we're called. If not, there are worse races in this
	// module.
	return (uintptr_t)pde_data(file_dentry(f)->d_parent->d_inode);
}
