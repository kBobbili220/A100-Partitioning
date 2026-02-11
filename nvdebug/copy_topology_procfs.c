// Copyright 2024 Joshua Bakita

#include "nvdebug_linux.h"

// Maximum number of LCEs that we will print
#define MAX_LCES 32

/* Which Logical Copy Engine (LCE) maps to a given Physical Copy Engine (PCE)?
  @param pce_id PCE index
  @return LCE index if mapping, -ENODEV on no mapping, and -errno otherwise
*/
int get_lce_for_pce(struct nvdebug_state *g, uint8_t pce_id) {
	int res;
	// LCEs only exist on Pascal+
	if (g->chip_id < NV_CHIP_ID_PASCAL)
		return -EOPNOTSUPP;

	if (g->chip_id < NV_CHIP_ID_VOLTA) {
		uint32_t config = nvdebug_readl(g, NV_LCE_FOR_PCE_GP100);
		if (config == -1)
			return -EIO;
		// On Pascal, two PCE configurations are packed per-byte.
		res = (config >> (pce_id * 4)) & 0xf;
		// 0x7 is the flag value for unconfigured on Pascal
		if (res == 0x7)
			return -ENODEV;
	} else if (g->chip_id < NV_CHIP_ID_AMPERE) {
		res = nvdebug_readl(g, NV_LCE_FOR_PCE_GV100(pce_id));
		// On the Titan V (GV100), bogus 0xbadf3000 observed if the GPU has yet to be
		// used since reset
		if (res == -1 || res == 0xbadf3000)
			return -EIO;
	} else {
		// Works through at least Ada
		res = nvdebug_readl(g, NV_LCE_FOR_PCE_GA100(pce_id));
		if (res == -1)
			return -EIO;
	}
	// At least on Volta through Ampere, 0xf is a flag value for unconfigured.
	if (res == 0xf)
		return -ENODEV;
	return res;
}

/* Which LCE does this GRaphics Copy Engine (GRCE) map to?
  @param grce_id GRCE index
  @return LCE index if mapping, -ENODEV on no mapping, and -errno otherwise
*/
int get_shared_lce_for_grce(struct nvdebug_state *g, uint8_t grce_id) {
	int res;
	uint32_t config;
	// LCEs only exist on Pascal+
	if (g->chip_id < NV_CHIP_ID_PASCAL)
		return -EOPNOTSUPP;

	if (g->chip_id < NV_CHIP_ID_VOLTA) {
		if ((config = nvdebug_readl(g, NV_GRCE_FOR_CE_GP100(0))) == -1)
			return -EIO;
		// One config per byte; bit 4 flags if shared
		if (((config >> (grce_id * 8)) & 0x8) == 0)
			return -ENODEV;
		// lower 3 bits contain the mapping
		res = (config >> (grce_id * 8)) & 0x7;
	} else if (g->chip_id < NV_CHIP_ID_AMPERE) {
		if ((config = nvdebug_readl(g, NV_GRCE_FOR_CE_GP100(grce_id))) == -1)
			return -EIO;
		// Only the lower 4 bits contain the mapping
		res = config & 0xf;
		if (res == 0xf)
			return -ENODEV;
	} else {
		// Works through at least Ada
		if ((config = nvdebug_readl(g, NV_GRCE_FOR_CE_GA100(grce_id))) == -1)
			return -EIO;
		// Only the lower 4 bits contain the mapping
		res = config & 0xf;
		if (res == 0xf)
			return -ENODEV;
	}
	return res;
}

typedef struct {
	enum {INVALID_CE, SHARED_LCE, PCE} type;
	uint8_t ce;
} lce2pce_entry_t;

/* Which PCE/LCE is each LCE mapped to?
  @param lce2pce     Array of lce2pce_entry_t to store mappings in
  @param lce2pce_len Number of array entries; at least 16 recommended
  @return -errno on error, 0 on success.
*/
int get_pces_for_lces(struct nvdebug_state *g, lce2pce_entry_t *lce2pce, int lce2pce_len) {
	uint32_t pce_id, grce_id, ce_pce_map;
	memset(lce2pce, INVALID_CE, lce2pce_len * sizeof(lce2pce_entry_t));

	if ((ce_pce_map = nvdebug_readl(g, NV_CE_PCE_MAP)) == -1)
		return -EIO;
	// Pull configuration for LCEs which directly map to a PCE
	for (pce_id = 0; pce_id < NV_CE_PCE_MAP_SIZE; pce_id++) {
		int lce;
		// Skip reading configuration if PCE is disabled
		if (((1 << pce_id) & ce_pce_map) == 0)
			continue;
		lce = get_lce_for_pce(g, pce_id);
		if (lce == -ENODEV)
			continue;
		if (lce < 0)
			return lce;
		if (lce > lce2pce_len)
			return -ERANGE;
		lce2pce[lce].type = PCE;
		lce2pce[lce].ce = pce_id;
	}
	// Pull configuration for LCEs which share a PCE with another LCE
	// GRCE0 is synonymous with LCE0 (GRCE1 and LCE1 likewise)
	// Only aware of up to two GRCEs per GPU
	for (grce_id = 0; grce_id < NV_GRCE_MAX; grce_id++) {
		int shared_lce;
		// GRCEs with a PCE already associated do not share with an LCE
		if (lce2pce[grce_id].type != INVALID_CE)
			continue;
		shared_lce = get_shared_lce_for_grce(g, grce_id);
		// Each GRCE should be associated with a PCE or shared LCE
		if (shared_lce == -ENODEV) {
			printk(KERN_WARNING "[nvdebug] GRCE%d unconfigured.\n", grce_id);
			continue;
		}
		if (shared_lce < 0)
			return shared_lce;
		lce2pce[grce_id].type = SHARED_LCE;
		lce2pce[grce_id].ce = shared_lce;
	}
	return 0;
}

typedef struct {
	int idx; // Index of LCE to print
	lce2pce_entry_t lce2pce[MAX_LCES]; // MAX_LCES-length table from get_pces_for_lces()
} copy_topology_iter_t;

// The *_seq_* functions in this file follow the patterns in
// device_info_procfs.c. See there for comments on implementation.
static void *copy_topology_file_seq_start(struct seq_file *s, loff_t *pos) {
	struct nvdebug_state *g = &g_nvdebug_state[file2parentgpuidx(s->file)];
	static copy_topology_iter_t iter;
	int err;
	if (*pos == 0)
		iter.idx = 0;
	if ((err = get_pces_for_lces(g, iter.lce2pce, MAX_LCES)) < 0)
		return ERR_PTR(err);
	if (iter.idx >= MAX_LCES)
		return NULL;
	return &iter;
}

static void* copy_topology_file_seq_next(struct seq_file *s, void *iter_raw,
                                         loff_t *pos) {
	copy_topology_iter_t *iter = (copy_topology_iter_t*)iter_raw;
	(*pos)++; // Required by seq interface
	if (++iter->idx >= MAX_LCES)
		return NULL;
	return iter;
}

static int copy_topology_file_seq_show(struct seq_file *s, void *iter_raw) {
	copy_topology_iter_t *iter = (copy_topology_iter_t*)iter_raw;
	lce2pce_entry_t entry = iter->lce2pce[iter->idx];
	if (entry.type == INVALID_CE)
		return 0;
	// First half: The LCE/GRCE in question
	if (iter->idx >= NV_GRCE_MAX)
		seq_printf(s, "LCE%02d -> ", iter->idx);
	else
		seq_printf(s, "GRCE%d -> ", iter->idx);
	// Second half: The PCE/LCE/GRCE that the LCE/GRCE in question is mapped to
	if (entry.type == PCE)
		seq_printf(s, "PCE%02d\n", entry.ce);
	else if (entry.ce >= NV_GRCE_MAX) // Shared LCE
		seq_printf(s, "LCE%02d\n", entry.ce);
	else // Shared GRCE
		seq_printf(s, "GRCE%d\n", entry.ce);
	return 0;
}

static void copy_topology_file_seq_stop(struct seq_file *s, void *lce2pce) {
	// No cleanup needed
}

static const struct seq_operations copy_topology_file_seq_ops = {
	.start = copy_topology_file_seq_start,
	.next = copy_topology_file_seq_next,
	.show = copy_topology_file_seq_show,
	.stop = copy_topology_file_seq_stop,
};

static int copy_topology_file_open(struct inode *inode, struct file *f) {
	return seq_open(f, &copy_topology_file_seq_ops);
}

struct file_operations copy_topology_file_ops = {
	.open = copy_topology_file_open,
	.read = seq_read,
	.llseek = seq_lseek,
	.release = seq_release,
};
