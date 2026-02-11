#include "nvdebug_linux.h"
#include <linux/seq_file.h> // For seq_* functions and types
#include <linux/uaccess.h> // For copy_to_user()

// Generic register printing function, used for PTOP_*_NUM registers (+more)
// @param f    File being read from. `data` field is register offset to read.
// @param buf  User buffer for result
// @param size Length of user buffer
// @param off  Requested offset. Updated by number of characters written.
// @return -errno on error, otherwise number of bytes written to *buf
// Note: Parent `data` field MUST be the GPU index
static ssize_t nvdebug_reg32_read(struct file *f, char __user *buf, size_t size, loff_t *off) {
	char out[16];
	int chars_written;
	uint32_t read;
	struct nvdebug_state *g = &g_nvdebug_state[file2parentgpuidx(f)];
	if (size < 16 || *off != 0)
		return 0;

	if ((read = nvdebug_readl(g, (uintptr_t)pde_data(file_inode(f)))) == -1)
		return -EIO;
	// 32 bit register will always take less than 16 characters to print
	chars_written = scnprintf(out, 16, "%#0x\n", read);
	if (copy_to_user(buf, out, chars_written))
		printk(KERN_WARNING "[nvdebug] %s: Unable to copy all data for %s\n", __func__, file_dentry(f)->d_name.name);
	*off += chars_written;
	return chars_written;
}

struct file_operations nvdebug_read_reg32_file_ops = {
	.read = nvdebug_reg32_read,
	.llseek = default_llseek,
};

typedef union {
	struct {
		uint8_t partitioning_select:2;
		uint8_t table_select:2;
		 uint32_t pad_1:12;
		uint8_t veid_offset:6;
		 uint32_t pad_2:2;
		uint8_t table_offset:6;
		 uint32_t pad_3:2;
	};
	uint32_t raw;
} partition_ctl_t;

static ssize_t nvdebug_read_part(struct file *f, char __user *buf, size_t size, loff_t *off) {
	char out[12*64+2];
	int i, chars_written = 0;
	partition_ctl_t part_ctl;
	struct nvdebug_state *g = &g_nvdebug_state[file2parentgpuidx(f)];
	if (size < 16 || *off != 0)
		return 0;
	// 32 bit register will always take less than 16 characters to print
	part_ctl.raw = nvdebug_readl(g, 0x00405b2c);
	//part_ctl.partitioning_select = 0; // XXX XXX XXX Temp; 06/18/2024
	//part_ctl.table_select = 3; // 3 == ???
	//part_ctl.table_select = 2; // 2 == TBL_SEL_PARTITIONING_LMEM_BLK
	part_ctl.table_select = 1; // 1 == TBL_SEL_PARTITIONING_ENABLE
	//part_ctl.table_select = 0; // 0 == TBL_SEL_NONE
	part_ctl.veid_offset = (uintptr_t)pde_data(file_inode(f)); // Range of [0, 0x3f], aka [0, 63]
	for (i = 0; i < 64; i++) {
		// Increment to next table offset in PARTITION_CTL
		part_ctl.table_offset = i;
		nvdebug_writel(g, 0x00405b2c, part_ctl.raw);
		// Verify write applied to PARTITION_CTL
		part_ctl.raw = nvdebug_readl(g, 0x00405b2c);
		if (part_ctl.table_offset != i)
			return -ENOTRECOVERABLE;
		// Read PARTITION_DATA and print
		// ---
		// I get back 0x000000ff on Volta and 0x00000003 on Turing from
		// PARTITION_DATA for all possible VEID_OFFSET, TBL_OFFSET, and TBL_SEL
		// combinations.
		// ---
		// There's a 48-byte (12-word) gap after the address for PARTITION_DATA.
		// Exploring this on Turing for TBL_SEL_PARTITIONING_ENABLE, VEID 1, 62, and
		// 63, with CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=5 for constant_cycles_kernel
		// running under MPS:
		// +0x0: 0x3
		// +0x4: 0
		// +0x8: 0x100
		// +0xC: 0
		// +0x10: 0xffffffff
		// +0x14: 0
		// +0x18: 0
		// +0x1C: 0xffffffff
		// +0x20: 0
		// +0x24: 0xffffffff
		// +0x28: 0xffffffff
		// +0x2C: 0xffffffff
		chars_written += scnprintf(out + chars_written, 12, "%#010x ", nvdebug_readl(g, 0x00405b30));
	}
	chars_written += scnprintf(out + chars_written, 2, "\n");
	if (copy_to_user(buf, out, chars_written))
		printk(KERN_WARNING "Unable to copy all data for %s\n", file_dentry(f)->d_name.name);
	*off += chars_written;
	return chars_written;
}

struct file_operations nvdebug_read_part_file_ops = {
	.read = nvdebug_read_part,
	.llseek = default_llseek,
};

static ssize_t nvdebug_reg_range_read(struct file *f, char __user *buf, size_t size, loff_t *off) {
	char out[12];
	int chars_written;
	uint32_t read, mask;
	struct nvdebug_state *g = &g_nvdebug_state[file2parentgpuidx(f)];
	// `start_bit` is included, `stop_bit` is not, so to print lower eight bits
	// from a register, use `start_bit = 0` and `stop_bit = 8`.
	union reg_range range;
	range.raw = (uintptr_t)pde_data(file_inode(f));

	// "0x" + up to 32-bit register as hex + "\n\0" is at most 12 characters
	if (size < 12 || *off != 0)
		return 0;

	// Print bits `start_bit` to `stop_bit` from 32 bits at address `offset`
	if ((read = nvdebug_readl(g, range.offset)) == -1)
		return -EIO;
	// Setup `mask` used to throw out unused upper bits
	mask = -1u >> (32 - range.stop_bit + range.start_bit);
	// Throw out unused lower bits via a shift, apply the mask, and print
	chars_written = scnprintf(out, 12, "%#0x\n", (read >> range.start_bit) & mask);
	if (copy_to_user(buf, out, chars_written))
		printk(KERN_WARNING "Unable to copy all data for %s\n", file_dentry(f)->d_name.name);
	*off += chars_written;
	return chars_written;
}

// Generic mechanism used for printing a subset of bits from a register
// Please store a `union reg_range` rather than a `uintptr_t` in the pde_data
struct file_operations nvdebug_read_reg_range_file_ops = {
	.read = nvdebug_reg_range_read,
	.llseek = default_llseek,
};

static ssize_t local_memory_read(struct file *f, char __user *buf, size_t size, loff_t *off) {
	struct nvdebug_state *g = &g_nvdebug_state[file2parentgpuidx(f)];
	char out[30];
	int chars_written;
	memory_range_t mem_range;
	if (size < 30 || *off != 0)
		return 0;
	mem_range.raw = nvdebug_readl(g, NV_FB_MMU_LOCAL_MEMORY_RANGE);
	if (mem_range.raw == -1)
		return -EIO;
	// 64-bit size has at most 19 characters + 8 for text and termination
	chars_written = scnprintf(out, 30, "%lld bytes\n", memory_range_to_bytes(mem_range));
	if (copy_to_user(buf, out, chars_written))
		printk(KERN_WARNING "Unable to copy all data for %s\n", file_dentry(f)->d_name.name);
	*off += chars_written;
	return chars_written;
}

// Read out size of on-device VRAM
struct file_operations local_memory_file_ops = {
	.read = local_memory_read,
	.llseek = default_llseek,
};

typedef struct {
	int idx; // Current index in the device_info table
	int length; // Length of device_info table (including unpopulated entries)
	int type_of_next_entry; // Only used on Ampere+ GPUs
	bool has_next_entry; // Only used on Ampere+ GPUs for show() idempotence
} device_info_iter;

//// ==v== PTOP_DEVICE_INFO ==v== ////

// Called to start or resume a sequence. Prior to 4.19, *pos is unreliable.
// Initializes iterator `iter` state and returns it. Ends sequence on NULL.
static void* device_info_file_seq_start(struct seq_file *s, loff_t *pos) {
	static device_info_iter iter;
	// If freshly starting a sequence, reset the iterator
	if (*pos == 0) {
		struct nvdebug_state *g = &g_nvdebug_state[seq2gpuidx(s)];
		iter.idx = 0;
		iter.type_of_next_entry = 0;
		iter.has_next_entry = 0;
		// On Ampere+, the device_info table length can vary
		if (g->chip_id >= NV_CHIP_ID_AMPERE)
			iter.length = NV_PTOP_DEVICE_INFO__SIZE_1_GA100(g);
		else
			iter.length = NV_PTOP_DEVICE_INFO__SIZE_1_GK104;
	}
	// Number of possible info entries is fixed, and list is sparse, so stop
	// iterating only when all entries have been checked, rather than on the first
	// empty entry.
	if (iter.idx >= iter.length)
		return NULL;
	return &iter;
}

// Steps to next record. Returns `&iter` (address should not change)
// Calls show() on non-NULL return
static void* device_info_file_seq_next(struct seq_file *s, void *iter_raw,
				       loff_t *pos) {
	device_info_iter *iter = (device_info_iter*)iter_raw;
	(*pos)++; // Required by seq interface
	// Number of possible info entries is fixed, and list is sparse, so stop
	// iterating only when all entries have been checked, rather than on the first
	// empty entry.
	if (++iter->idx >= iter->length)
		return NULL;
	// The info_type field is not available in the Ampere device_info data, so
	// it must be inferred. NOP for older devices (cheaper than another branch).
	// This has to be here (rather than in the show function) to support the
	// idempotence requirements of show() in the seq_file interface.
	iter->type_of_next_entry = iter->has_next_entry ? iter->type_of_next_entry + 1 : 0;
	return iter;
}

// Print info at iter->idx for Kepler--Turing GPUs. Returns non-zero on error.
// Implementation of this function must be idempotent
static int device_info_file_seq_show_gk104(struct seq_file *s, void *iter_raw) {
	device_info_iter *iter = (device_info_iter*)iter_raw;
	ptop_device_info_gk104_t curr_info;
	struct nvdebug_state *g = &g_nvdebug_state[seq2gpuidx(s)];
	curr_info.raw = nvdebug_readl(g, NV_PTOP_DEVICE_INFO_GK104(iter->idx));
	// Check for read errors
	if (curr_info.raw == -1)
		return -EIO;

	// Parse and print the data
	switch(curr_info.info_type) {
	case INFO_TYPE_DATA:
		// As of early 2022, only the ENUM2 format of this entry exists
		if (curr_info.is_not_enum2)
			break;
		seq_printf(s, "| BAR0 Base %#.8x\n"
			      "|           instance %d\n",
			curr_info.pri_base << 12, curr_info.inst_id);
		if (curr_info.fault_id_is_valid)
			seq_printf(s, "| Fault ID:        %3d\n", curr_info.fault_id);
		break;
	case INFO_TYPE_ENUM:
		if (curr_info.engine_is_valid)
			seq_printf(s, "| Host's Engine ID: %2d\n", curr_info.engine_enum);
		if (curr_info.runlist_is_valid)
			seq_printf(s, "| Runlist ID:       %2d\n", curr_info.runlist_enum);
		if (curr_info.intr_is_valid)
			seq_printf(s, "| Interrupt ID:     %2d\n", curr_info.intr_enum);
		if (curr_info.reset_is_valid)
			seq_printf(s, "| Reset ID:         %2d\n", curr_info.reset_enum);
		break;
	case INFO_TYPE_ENGINE_TYPE:
		seq_printf(s, "| Engine Type:      %2d (", curr_info.engine_type);
		if (curr_info.engine_type < ENGINE_TYPES_LEN)
			seq_printf(s, "%s)\n", ENGINE_TYPES_NAMES[curr_info.engine_type]);
		else
			seq_printf(s, "Unknown Historical)\n");
		break;
	case INFO_TYPE_NOT_VALID:
	default:
		// Device info records are sparse, so skip unset or unknown ones
		return 0;
	}

	// Draw a line between each device entry
	if (!curr_info.has_next_entry)
		seq_printf(s, "+---------------------+\n");
	return 0;
}

// Print info at iter->idx for Ampere+ GPUs. Returns non-zero on error.
// Implementation of this function must be idempotent
static int device_info_file_seq_show_ga100(struct seq_file *s, void *iter_raw) {
	device_info_iter *iter = (device_info_iter*)iter_raw;
	ptop_device_info_ga100_t curr_info;
	struct nvdebug_state *g = &g_nvdebug_state[seq2gpuidx(s)];
	curr_info.raw = nvdebug_readl(g, NV_PTOP_DEVICE_INFO_GA100(iter->idx));
	// Check for read errors
	if (curr_info.raw == -1)
		return -EIO;

	// Update tracking data only used by next(); allows preserving idempotence
	iter->has_next_entry = curr_info.has_next_entry;
	// Silently skip empty entries
	if (curr_info.raw == 0)
		return 0;
	// In nvdebug, an entry is considered invalid if it does not consist of at
	// least two rows. So, if this is the first row of an entry, but another row
	// is not indicated, this entry is invalid and should be skipped.
	if (iter->type_of_next_entry == 0 && !curr_info.has_next_entry) {
		printk(KERN_WARNING "[nvdebug] Skipping seemingly-invalid device_info entry (idx: %d, raw: %#0x)\n", iter->idx, curr_info.raw);
		return 0;
	}

	// Parse and print the data
	// Note: The goal of this interface is to present useful information to
	// a human user, NOT to provide a stable format for scripts to parse.
	// Because of this, we favor accurately printing the data in each entry,
	// rather than providing stable (if imperfectly correct) field names
	switch(iter->type_of_next_entry) {
	case 0:
		seq_printf(s, "| Engine Type:     %3d (", curr_info.engine_type);
		if (curr_info.engine_type < ENGINE_TYPES_LEN)
			seq_printf(s, "%s)\n", ENGINE_TYPES_NAMES[curr_info.engine_type]);
		else
			seq_printf(s, "Unknown, introduced post-Lovelace)\n");
		seq_printf(s, "|           instance %d\n", curr_info.inst_id);
		seq_printf(s, "| Fault ID:       %4d\n", curr_info.fault_id);
		break;
	case 1:
		seq_printf(s, "| BAR0 Base %#.8x\n", curr_info.pri_base << 8);
		seq_printf(s, "| Reset ID:        %3d\n", curr_info.reset_id);
		seq_printf(s, "| Is Engine:         %1d\n", curr_info.is_engine);

		break;
	case 2:
		seq_printf(s, "| Runlist Eng. ID:   %1d\n", curr_info.rleng_id);
		// Theoretically, we could extract an ID from the runlist RAM
		seq_printf(s, "| RL Base:  %#.8x\n", curr_info.runlist_pri_base << 10);
		break;
	default:
		printk(KERN_WARNING "[nvdebug] Skipping unexpected continuation of device_info entry (idx: %d, raw: %#0x)\n", iter->idx, curr_info.raw);
	}

	// Draw a line between each device entry
	if (!curr_info.has_next_entry)
		seq_printf(s, "+---------------------+\n");
	return 0;
}

static void device_info_file_seq_stop(struct seq_file *s, void *idx) {
	// No cleanup needed
}

static struct seq_operations device_info_file_seq_ops = {
	.start = device_info_file_seq_start,
	.next = device_info_file_seq_next,
	.stop = device_info_file_seq_stop,
};

static int device_info_file_open(struct inode *inode, struct file *f) {
	if (g_nvdebug_state[file2parentgpuidx(f)].chip_id >= NV_CHIP_ID_AMPERE)
		device_info_file_seq_ops.show = device_info_file_seq_show_ga100;
	else
		device_info_file_seq_ops.show = device_info_file_seq_show_gk104;
	return seq_open(f, &device_info_file_seq_ops);
}

struct file_operations device_info_file_ops = {
	.open = device_info_file_open,
	.read = seq_read,
	.llseek = seq_lseek,
	.release = seq_release,
};
