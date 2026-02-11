/* Copyright 2024 Joshua Bakita
 * Helpers to deal with NVIDIA's MMU and associated page tables
 */
#include <linux/dma-mapping.h>  // dma_map_page() and dma_unmap_page()
#include <linux/err.h>  // ERR_PTR() etc.
#include <linux/gfp.h>  // alloc_pages()
#include <linux/iommu.h>  // iommu_get_domain_for_dev() and iommu_iova_to_phys()
#include <linux/kernel.h>  // Kernel types
#include <linux/list.h>  // struct list_head and associated functions
#include <linux/mm.h>  // put_page()

#include "nvdebug.h"

/* Set logging level for MMU operations
  g_verbose >= 1: Log a single message describing the MMU operation
  g_verbose >= 2: Log every PDE and PTE traversed
*/
int g_verbose = 0;
#define printk_debug if (g_verbose >= 2) printk
#define printk_info  if (g_verbose >= 1) printk

// At least map_page_directory() assumes that pages are 4 KiB
#if PAGE_SIZE != 4096
#error nvdebug assumes and requires a 4 KiB page size.
#endif

/* Convert a page directory (PD) pointer and aperture to be kernel-accessible

  I/O MMU handling inspired by amdgpu_iomem_read() in amdgpu_ttm.c of the
  AMDGPU driver.

  @param addr  Pointer from page directory entry (PDE)
  @param pd_ap PD-type aperture (target address space) for `addr`
  @return A dereferencable kernel address, 0 if an I/O MMU is in use and has
          no available mapping for the bus address, or an ERR_PTR-wrapped error
 */
static void __iomem *pd_deref(struct nvdebug_state *g, uintptr_t addr,
                              enum PD_TARGET pd_ap) {
	struct iommu_domain *dom;
	phys_addr_t phys;

	// Validate arguments
	if (unlikely(!IS_PD_TARGET(pd_ap) || pd_ap == PD_AND_TARGET_INVALID || !addr))
		return ERR_PTR(-EINVAL);

	// VID_MEM accesses are the simple common-case
	if (pd_ap == PD_AND_TARGET_VID_MEM) {
		// Using BAR2 requires a page-table traversal. As this function is part
		// of the page-table traversal process, it must instead use PRAMIN.
		int off = addr_to_pramin_mut(g, addr, TARGET_VID_MEM);
		if (off < 0)
			return ERR_PTR(off);
		return g->regs + NV_PRAMIN + off;
	}
	/* SYS_MEM accesses are rare. Only nvgpu (Jetson driver), nouveau, and this
	 * driver are known to create page directory entries in SYS_MEM.
	 *
	 * On systems using an I/O MMU, or some other I/O virtual address space,
	 * these are **not** physical addresses, and must first be translated
	 * through the I/O MMU before use.
	 * Example default meaning of a SYS_MEM address for a few CPUs:
	 * - Jetson Xavier : physical address
	 * - AMD 3950X     : I/O MMU address
	 * - Phenom II x4  : physical address
	 */
	// Check for, and translate through, the I/O MMU (if any)
	if ((dom = iommu_get_domain_for_dev(g->dev))) {
		phys = iommu_iova_to_phys(dom, addr);
		printk_debug(KERN_DEBUG "[nvdebug] %s: I/O MMU translated SYS_MEM I/O VA %#lx to physical address %#llx.\n", __func__, addr, phys);
	} else
		phys = addr;

	if (!phys)
		return 0;

	return phys_to_virt(phys);
}

// Internal helper for search_page_directory().
uint64_t search_page_directory_subtree(struct nvdebug_state *g,
                                       uintptr_t pde_addr,
                                       enum PD_TARGET pde_target,
                                       uint64_t addr_to_find,
                                       enum INST_TARGET addr_to_find_aperture,
                                       uint32_t level) {
	uint64_t res, i;
	void __iomem *pde_kern;
	page_dir_entry_t entry;
	if (level > sizeof(NV_MMU_PT_V2_SZ))
		return 0;
	// Hack to workaround PDE0 being double-size and strangely formatted
	if (NV_MMU_PT_V2_ENTRY_SZ[level] == 16)
		pde_addr += 8;
	// Translate a VID_MEM/SYS_MEM-space address to something kernel-accessible
	pde_kern = pd_deref(g, pde_addr, pde_target);
	if (IS_ERR_OR_NULL(pde_kern)) {
		printk(KERN_ERR "[nvdebug] %s: Unable to resolve %#lx in GPU %s to a kernel-accessible address. Error %ld.\n", __func__, pde_addr, pd_target_to_text(pde_target), PTR_ERR(pde_kern));
		return 0;
	}
	// Read the page directory entry (a pointer to another directory, or a PTE)
	entry.raw_w = readq(pde_kern);
	// If we reached an invalid (unpopulated) PDE, walk back up the tree
	if (entry.target == PD_AND_TARGET_INVALID)
		return 0;
	// Succeed when we reach a PTE with the address we want
	if (entry.is_pte) {
		// TODO: Handle huge pages here
		printk_debug(KERN_DEBUG "[nvdebug] PTE for phy addr %#018llx, ap '%s', vol '%d', priv '%d', ro '%d', no_atomics '%d' (raw: %#018llx)\n", ((u64)entry.addr_w) << 12, pd_target_to_text(entry.target), entry.is_volatile, entry.is_privileged, entry.is_readonly, entry.atomics_disabled, entry.raw_w);
		return (uint64_t)entry.addr << 12 == addr_to_find && entry.aperture == addr_to_find_aperture;
	}
	printk_debug(KERN_DEBUG "[nvdebug] Found PDE pointing to %#018llx in ap '%s' vol '%d' at lvl %d (raw: %#018llx)\n", ((u64)entry.addr_w) << 12, pd_target_to_text(entry.target), entry.is_volatile, level, entry.raw_w);
	// Depth-first search of the page table
	for (i = 0; i < NV_MMU_PT_V2_SZ[level + 1]; i++) {
		uint64_t next = ((uint64_t)entry.addr << 12) + NV_MMU_PT_V2_ENTRY_SZ[level + 1] * i;
		printk_debug(KERN_DEBUG "[nvdebug] Searching index %llu in lvl %d\n", i, level + 1);
		res = search_page_directory_subtree(g, next, entry.target, addr_to_find, addr_to_find_aperture, level + 1);
		if (res)
			return res | (i << NV_MMU_PT_V2_LSB[level + 1]);
	}
	return 0;
}

/* GPU Physical address -> Virtual address ("reverse" translation) for V2 tables
  Depth-first search a page directory of the GPU MMU for where a particular
  physical address is mapped. Upon finding a mapping, the virtual address is
  returned.

  The page directory and tables may be located in VID_MEM, SYS_MEM, or spread
  across multiple apertures.

  @param pd_config    Page Directory configuration, containing pointer and
                      aperture for the start of the PDE3 entries
  @param addr_to_find Physical address to reconstruct the virtual address of
  @param addr_to_find_aperture Aperture (SYS_MEM or VID_MEM) of addr_to_find
  @return 0 on error, otherwise the virtual address at which addr_to_find is
          mapped into by this page table. (Zero is not a valid virtual address)
*/
uint64_t search_page_directory(struct nvdebug_state *g,
                               page_dir_config_t pd_config,
                               uint64_t addr_to_find,
                               enum INST_TARGET addr_to_find_aperture) {
	uint64_t res, i;
	// Make sure that the query is page-aligned
	if (addr_to_find & 0xfff) {
		printk(KERN_WARNING "[nvdebug] Attempting to search for unaligned address %llx in search_page_directory()!\n", addr_to_find);
		return 0;
	}
	printk_info(KERN_INFO "[nvdebug] Searching for addr %#018llx in page table with base %#018lx\n", addr_to_find, (uintptr_t)pd_config.page_dir << 12);
	// Search the top-level page directory (PDE3)
	for (i = 0; i < NV_MMU_PT_V2_SZ[0]; i++)
		if ((res = search_page_directory_subtree(g, ((uintptr_t)pd_config.page_dir << 12) + NV_MMU_PT_V2_ENTRY_SZ[0] * i, INST2PD_TARGET(pd_config.target), addr_to_find, addr_to_find_aperture, 0)))
			return (res & ~0xfff) | (i << NV_MMU_PT_V2_LSB[0]);
	return 0;
}

/* GPU Virtual address -> Physical address ("forward" translation) for V2 tables
  Index the page directories and tables used by the GPU MMU to determine which
  physical address a given GPU virtual address has been mapped to.

  The page directory and tables may be located in VID_MEM, SYS_MEM, or spread
  across multiple apertures.

  @param pd_config      Page Directory configuration, containing pointer and
                        aperture for the start of the PDE3 entries
  @param addr_to_find   Virtual address to translate to a physical address
  @param found_addr     Where to store found physical address (0 if unfound)
  @param found_aperture Where to store aperture of found physical address
  @return 0 on success, -ENXIO if not found, and -errno on error.
*/
int translate_page_directory(struct nvdebug_state *g,
                             page_dir_config_t pd_config,
                             uint64_t addr_to_find,
                             uint64_t *found_addr /* out */,
                             enum INST_TARGET *found_aperture /* out */) {
	page_dir_entry_t entry;
	void __iomem *next_kva;
	unsigned int level, pde_idx;
	uintptr_t next = (uintptr_t)pd_config.page_dir << 12;
	enum PD_TARGET next_target = INST2PD_TARGET(pd_config.target);

	*found_addr = 0;
	*found_aperture = TARGET_INVALID;

	// Make sure that the query is page-aligned (likely mistake otherwise)
	if (addr_to_find & 0xfff) {
		printk(KERN_WARNING "[nvdebug] Attempting to translate unaligned address %#llx in translate_page_directory()!\n", addr_to_find);
		return -EINVAL;
	}

	printk_info(KERN_INFO "[nvdebug] Translating addr %#018llx in V2 page table with base %#018llx\n", (u64)addr_to_find, (u64)next);

	// Step through each PDE level and the PTE level
	for (level = 0; level < 5; level++) {
		// Index into this level
		pde_idx = (addr_to_find >> NV_MMU_PT_V2_LSB[level]) & (NV_MMU_PT_V2_SZ[level] - 1);
		printk_debug(KERN_DEBUG "[nvdebug] Using index %u in lvl %d\n", pde_idx, level);
		// Hack to workaround PDE0 being double-size and strangely formatted
		if (NV_MMU_PT_V2_ENTRY_SZ[level] == 16)
			next += 8;
		// Obtain a kernel-dereferencable address
		next_kva = pd_deref(g, next, next_target);
		if (IS_ERR_OR_NULL(next_kva)) {
			printk(KERN_ERR "[nvdebug] %s: Unable to resolve %#lx in GPU %s to a kernel-accessible address. Error %ld.\n", __func__, next, pd_target_to_text(next_target), PTR_ERR(next_kva));
			return PTR_ERR(next_kva);
		}
		// Obtain entry at this level
		entry.raw_w = readq(next_kva + NV_MMU_PT_V2_ENTRY_SZ[level] * pde_idx);
		if (entry.target == PD_AND_TARGET_INVALID)
			return -ENXIO;
		printk_debug(KERN_DEBUG "[nvdebug] Found %s pointing to %#018llx in ap '%s' at lvl %d (raw: %#018llx)\n", entry.is_pte ? "PTE" : "PDE", ((u64)entry.addr) << 12, pd_target_to_text(entry.target), level, entry.raw_w);
		// Just return the physical address if this is the PTE level
		if (entry.is_pte) { // level == 4 for 4 KiB pages, == 3 for 2 MiB
			*found_addr = ((uint64_t)entry.addr) << 12;
			*found_aperture = entry.aperture;
			return 0;
		}
		// Otherwise step to the next table level
		// TODO: Use addr_w as appropriate
		next = (uint64_t)entry.addr << 12;
		next_target = entry.target;
	}

	return 0;
}

// This struct is very special. We will never directly allocate this struct;
// its sole purpose is to provide more intuitive names to the offsets at which
// we store data in Linux's struct page. Such (ab)use of struct page is
// explictly permitted (see linux/mm_types.h). This struct is thus used by
// casting a pointer of struct page to a pointer of struct nvdebug_pd_page,
// then accessing the associated fields. This pointer may also be freely cast
// back to a struct page pointer.
// We have 20 (32-bit) or 40 (64-bit) bytes available in the page struct
// (according to the documentation on struct page). We use 20 (32-bit) or 28
// (64-bit) bytes. Our comments indicate what available parts of struct page we
// repurpose for our own needs.
struct nvdebug_pd_page {
	unsigned long __flags; // From struct page; do not touch!
	// Overlaps struct page.lru
	struct list_head list; // 4/8 bytes
	// Overlaps struct page.lru (and page.mapping on 32-bit)
	uint64_t parent_addr; // 8 bytes
	// Overlaps struct page.mapping (page.share on 32-bit)
	enum PD_TARGET parent_aperture; // 4 bytes
	// Overlaps page.mapping and page.share (page.private on 32-bit)
	dma_addr_t dma_addr; // 4/8 bytes
};

/* Collect and free any now-unused page directory/table allocations

  @param force Deallocate all page directories/tables created by this module,
               no matter if they appear to be in-use or not.
  @returns Number of freed pages on success, -errno on error.
*/
int gc_page_directory(struct nvdebug_state *g, bool force) {
	struct nvdebug_pd_page  *page, *_page;
	void __iomem *parent_kva;
	page_dir_entry_t parent_entry;
	int freed_pages = 0;

	// Depth-first traversal (from perspective of each page table) of page
	// allocations.
	// (This is depth-first because map_page_directory() always allocates and
	// pushes page directory allocations before page table allocations.)
	list_for_each_entry_safe_reverse(page, _page, &g->pd_allocs, list) {
		printk_debug(KERN_DEBUG "[nvdebug] %s: Checking if page directory/table at %llx (SYS_MEM_?) with parent at %llx (%s) is unused...\n", __func__, page->dma_addr, page->parent_addr, pd_target_to_text(page->parent_aperture));
		// Try to determine if we're still in-use. We consider ourselves
		// potentially in-use if our parent still points to us.
		parent_kva = pd_deref(g, page->parent_addr, page->parent_aperture);
		if (IS_ERR(parent_kva)) {
			printk(KERN_ERR "[nvdebug] %s: Error resolving %#llx in GPU %s to a kernel-accessible address. Error %ld.\n", __func__, page->parent_addr, pd_target_to_text(page->parent_aperture), PTR_ERR(parent_kva));
			return -ENOTRECOVERABLE;
		}
		// A NULL kva indicates parent no longer exists
		parent_entry.raw_w = parent_kva ? readq(parent_kva) : 0;
		// Page directory/table still in-use; do not free unless forced
		if (parent_entry.addr_w == (page->dma_addr >> 12) && !force)
			continue;
		// Free this page table/directory and delete our parent's pointer to us
		if (parent_entry.addr_w == (page->dma_addr >> 12)) {
			printk(KERN_WARNING "[nvdebug] %s: Deleting page table/directory at %llx (SYS_MEM_?) with parent at %llx (%s) that may still be in-use!\n", __func__, page->dma_addr, page->parent_addr, pd_target_to_text(page->parent_aperture));
			writeq(0, parent_kva);
		}
		// Unmap, zero, free, and remove from tracking (these all return void)
		dma_unmap_page(g->dev, page->dma_addr, PAGE_SIZE, DMA_TO_DEVICE);
		memset(page_to_virt((struct page*)page), 0, PAGE_SIZE);
		// Same reset needed for mapping
		((struct page*)page)->mapping = NULL;
		// Remove this page from our list of allocated pages
		list_del(&page->list);
		// Free the page
		put_page((struct page*)page);
		freed_pages++;
	}
	printk_debug(KERN_DEBUG "[nvdebug] %s: Freed %d pages.", __func__, freed_pages);
	return freed_pages;
}

/* Map a GPU virtual address to a physical address in a GPU page table
  Search for a mapping for specified GPU virtual address, and create a new one
  if none is found. Automatically creates page directories and page table
  entries as necessary.

  The page directory and tables may be located in VID_MEM, SYS_MEM, or spread
  across multiple apertures.

  @param pd_config      Page Directory configuration, containing pointer and
                        aperture for the start of the PDE3 entries
  @param vaddr_to_find  Virtual address to check, and map to a physical address
                        if nothing is already mapped (up to 49 bits long)
  @param paddr_to_map   Physical address to use (up to 36 bits long if VID_MEM,
                        and up to 58 bits if SYS_MEM)
  @param paddr_target   Which space does the physical address refer to?
  @param huge_page      Set to map a 2 MiB, rather than 4 KiB, page
  @return 0 on success, 1 if mapping already exists, -EADDRINUSE if virtual
          address is already mapped to something else, and -errno on error
*/
int map_page_directory(struct nvdebug_state *g,
                       page_dir_config_t pd_config,
                       uint64_t vaddr_to_find,
                       uint64_t paddr_to_map,
                       enum INST_TARGET paddr_target,
                       bool huge_page) {
	page_dir_entry_t entry;
	void __iomem *next_kva;
	unsigned int level, pde_idx;
	uintptr_t next = (uintptr_t)pd_config.page_dir << 12;
	enum PD_TARGET next_target = INST2PD_TARGET(pd_config.target);

	// Make sure that the query is page-aligned (likely mistake otherwise)
	if ((vaddr_to_find & 0xfff || paddr_to_map & 0xfff)
	    || (huge_page && (vaddr_to_find & 0x1fffff || paddr_to_map & 0x1fffff))) {
		printk(KERN_WARNING "[nvdebug] %s: Attempting to map an unaligned address (physical %#018llx or virtual %#018llx)! Failing...\n", __func__, paddr_to_map, vaddr_to_find);
		return -EINVAL;
	}

	// NVIDIA supports up to 49-bit virtual addresss
	// Except Jetson Xavier only seems to be able to resolve 47-bit addresses?
	if (vaddr_to_find >> 49) {
		printk(KERN_WARNING "[nvdebug] %s: vaddr_to_find (%#018llx) is beyond the 49-bit virtual address space supported by the GPU! Failing...\n", __func__, vaddr_to_find);
		return -EINVAL;
	}

	// NVIDIA supports up to 36-bit VID_MEM addresses
	if (paddr_target == TARGET_VID_MEM && paddr_to_map >> 36) {
		printk(KERN_WARNING "[nvdebug] %s: paddr_to_map (%#018llx) is beyond the 36-bit VID_MEM address space! Failing...\n", __func__, paddr_to_map);
		return -EINVAL;
	}

	// NVIDIA supports up to 58-bit SYS_MEM addresses
	if ((paddr_target == TARGET_SYS_MEM_COHERENT ||
	     paddr_target == TARGET_SYS_MEM_NONCOHERENT) && paddr_to_map >> 58) {
		printk(KERN_WARNING "[nvdebug] %s: paddr_to_map (%#018llx) is beyond the 58-bit SYS_MEM address space! Failing...\n", __func__, paddr_to_map);
		return -EINVAL;
	}

	// We don't support mapping to PEERs; that requires a PEER ID
	if (paddr_target == TARGET_PEER) {
		printk(KERN_WARNING "[nvdebug] %s: paddr_target must be SYS_MEM_* or VID_MEM! Failing...\n", __func__);
		return -EINVAL;
	}

	printk_info(KERN_INFO "[nvdebug] Mapping addr %#018llx in page table with base %#018llx to %s address %#018llx\n", vaddr_to_find, (u64)next, target_to_text(paddr_target), paddr_to_map);

	// Step through each PDE level and the PTE level
	for (level = 0; level < 5; level++) {
		// Index into this level
		pde_idx = (vaddr_to_find >> NV_MMU_PT_V2_LSB[level]) & (NV_MMU_PT_V2_SZ[level] - 1);
		printk_debug(KERN_DEBUG "[nvdebug] In table at KVA %#lx, using index %u in lvl %d\n", (uintptr_t)next, pde_idx, level);
		// Hack to workaround PDE0 being double-size and strangely formatted
		if (NV_MMU_PT_V2_ENTRY_SZ[level] == 16)
			next += 8;
		// Obtain a kernel-dereferencable address
		next_kva = pd_deref(g, next, next_target);
		if (IS_ERR_OR_NULL(next_kva)) {
			printk(KERN_ERR "[nvdebug] %s: Unable to resolve %#lx in GPU %s to a kernel-accessible address. Error %ld.\n", __func__, next, pd_target_to_text(next_target), PTR_ERR(next_kva));
			return -ENOTRECOVERABLE;
		}
		// Obtain entry at this level
		entry.raw_w = readq(next_kva + NV_MMU_PT_V2_ENTRY_SZ[level] * pde_idx);
		// If pointer to next level of the table does not exist
		if (entry.target == PD_AND_TARGET_INVALID) { // PTE or PD covered by PD_AND_TARGET_INVALID
			if (level == 4 || (huge_page && level == 3)) {
				// Create new PTE (allocation, as needed, is handled at level 2 or 3)
				// Targets observed in page tables:
				// For PCIe: entry.target == PTE_AND_TARGET_VID_MEM;
				// For Jetson: entry.target == PTE_AND_TARGET_SYS_MEM_NONCOHERENT;
				entry.is_pte = 1;
				entry.aperture = paddr_target;
				if (paddr_target == TARGET_VID_MEM)
					entry.addr = paddr_to_map >> 12;
				else
					entry.addr_w = paddr_to_map >> 12;
				// Set the volatile bit (as NVRM does for SYS_MEM_COHERENT mappings)
				// (This does nothing if the target is VID_MEM, but if the target is
				// SYS_MEM_*, accesses will bypass the L2.)
				entry.is_volatile = 1;
				// Leave other fields zero, yielding an unencrypted, unprivileged, r/w,
				// volatile mapping with atomics enabled.

				// XXX: Hack to work around PDE0 double-size weirdness. Huge
				//      page mapping will fault without this.
				if (level == 3)
					writeq(entry.raw_w, next_kva - 8 + NV_MMU_PT_V2_ENTRY_SZ[level] * pde_idx);
			} else {
				struct page* page_dir;
				struct nvdebug_pd_page* page_dir_reinterpret;
				dma_addr_t page_dir_dma;
				// Allocate one 4 KiB all-zero (all invalid) page directory/
				// table at the next level
				if (!(page_dir = alloc_pages(GFP_KERNEL | __GFP_ZERO, 0)))
					return -ENOMEM;
				// Obtain a GPU-accessible/bus address for this page (handling
				// I/O MMU mappings, etc.)
				page_dir_dma = dma_map_page(g->dev, page_dir, 0, PAGE_SIZE, DMA_TO_DEVICE);
				// Verify that we were able to create a mapping
				if (dma_mapping_error(g->dev, page_dir_dma))
					return dma_mapping_error(g->dev, page_dir_dma);
				// Record this allocation for freeing later
				// Note: Linux maintains a page struct for every page in the
				//       system. This struct has available space that drivers
				//       can use to store their own tracking information. Our
				//       struct nvdebug_pd_page facilitates this.
				page_dir_reinterpret = (struct nvdebug_pd_page*)page_dir;
				page_dir_reinterpret->parent_addr = next + NV_MMU_PT_V2_ENTRY_SZ[level] * pde_idx;
				page_dir_reinterpret->parent_aperture = next_target;
				page_dir_reinterpret->dma_addr = page_dir_dma;
				list_add(&page_dir_reinterpret->list, &g->pd_allocs);
				// Point this entry to the new directory/table
				entry.target = PD_AND_TARGET_SYS_MEM_COHERENT; // Observed in page tables
				// Must use addr_w with SYS_MEM targets
				entry.addr_w = page_dir_dma >> 12;
				// On Jetson and NVRM, all PDEs are marked volatile
				entry.is_volatile = 1;
				// We don't configure ATS, so disable ATS lookups for speed.
				entry.no_ats = 1;
			}
			writeq(entry.raw_w, next_kva + NV_MMU_PT_V2_ENTRY_SZ[level] * pde_idx);
			printk_debug(KERN_DEBUG "[nvdebug] Created %s pointing to %llx in ap '%s' at lvl %d (raw: %#018llx)\n", entry.is_pte ? "PTE" : "PDE", ((u64)entry.addr) << 12, pd_target_to_text(entry.target), level, entry.raw_w);
			// Successfully created the requested PTE, so return
			if (entry.is_pte)
				return 0;
		} else {
			printk_debug(KERN_DEBUG "[nvdebug] Found %s pointing to %llx in ap '%s' at lvl %d (raw: %#018llx)\n", entry.is_pte ? "PTE" : "PDE", ((u64)entry.addr) << 12, pd_target_to_text(entry.target), level, entry.raw_w);
		}

		// If this is the PTE level, return success if the address and target are correct
		if (entry.is_pte) { // level == 4 for 4 KiB pages, == 3 for 2 MiB
			if (entry.aperture != paddr_target)
				return -EADDRINUSE; // Also handles PEER
			if (entry.aperture == TARGET_VID_MEM)
				return (uint64_t)entry.addr == paddr_to_map >> 12 ? 1 : -EADDRINUSE;
			else
				return entry.addr_w == paddr_to_map >> 12 ? 1 : -EADDRINUSE; // SYS_MEM is wider
		}

		// If mapping a 2 MiB page and we made it here, level 3 had a PDE. This
		// means that the requested 2 MiB virtual region already has one or more
		// small pages mapped within it---a.k.a., the addresses are in use.
		// If we didn't bail out here, the above logic would attempt to fallback
		// to a 4 KiB mapping, which would be unexpected behavior.
		if (huge_page && level == 3)
			return -EADDRINUSE;

		// Otherwise step to the next table level
		if (entry.aperture == TARGET_VID_MEM)
			next = (uint64_t)entry.addr << 12;
		else
			next = (uint64_t)entry.addr_w << 12; // SYS_MEM is wider
		next_target = entry.target;
	}

	return -ENOTRECOVERABLE; // Should be impossible
}

/* GPU Physical address -> Virtual address ("reverse" translation) for V1 tables
  (See `search_page_directory()` for documentation.)
 */
uint64_t search_v1_page_directory(struct nvdebug_state *g,
                                  page_dir_config_t pd_config,
                                  uint64_t addr_to_find,
                                  enum INST_TARGET addr_to_find_aperture) {
	uint64_t j, i = 0;
	page_dir_entry_v1_t pde;
	page_tbl_entry_v1_t pte;
	uintptr_t pte_offset, pde_offset;
	void __iomem *pte_addr, *pde_addr;

	// This function only understands the Page Table Version 1 format
	if (pd_config.is_ver2) {
		printk(KERN_ERR "[nvdebug] Passed a Version 2 page table at %#018llx to translate_v1_page_directory()!\n", (uint64_t)pd_config.page_dir << 12);
		return 0;
	}

	// We only understand the Version 1 format when 128 KiB huge pages are in-use
	if (pd_config.is_64k_big_page) {
		printk(KERN_ERR "[nvdebug] Page Table Version 1 with 64 KiB huge pages is unsupported!\n");
		return 0;
	}

	printk_info(KERN_INFO "[nvdebug] Searching V1 page table at %#018lx in %s for addr %#018llx\n", (uintptr_t)pd_config.page_dir << 12, target_to_text(pd_config.target), addr_to_find);

	// For each PDE
	do {
		// Index the list of page directory entries
		pde_offset = ((uint64_t)pd_config.page_dir << 12) + i * sizeof(page_dir_entry_v1_t);
		// Convert the VID_MEM/SYS_MEM address to a kernel-accessible addr
		pde_addr = pd_deref(g, pde_offset, INST2PD_TARGET(pd_config.target));
		if (IS_ERR_OR_NULL(pde_addr)) {
			printk(KERN_ERR "[nvdebug] %s: Unable to resolve %#lx in GPU %s to a kernel-accessible address. Error %ld.\n", __func__, pde_offset, pd_target_to_text(INST2PD_TARGET(pd_config.target)), -PTR_ERR(pde_addr));
			return 0;
		}
		// readq doesn't seem to work on BAR0
		pde.raw = readl(pde_addr + 4);
		pde.raw <<= 32;
		pde.raw |= readl(pde_addr);
		// Verify PDE is present
		if (pde.target == PD_TARGET_INVALID && pde.alt_target == PD_TARGET_INVALID)
			continue;
		// TODO: Handle huge pages
		printk_debug(KERN_DEBUG "[nvdebug] Found %s PDE at index %lld pointing to PTEs @ %#018llx in ap '%d' (raw: %#018llx)\n", pde.alt_is_volatile ? "volatile" : "non-volatile", i, ((u64)pde.alt_addr) << 12, pde.alt_target, pde.raw);
		// For each PTE
		for (j = 0; j < NV_MMU_PT_V1_SZ[1]; j++) {
			// Index the list of page table entries starting at pde.alt_addr
			pte_offset = ((uint64_t)pde.alt_addr << 12) + j * sizeof(page_tbl_entry_v1_t);
			// Convert the VID_MEM/SYS_MEM address to a kernel-accessible addr
			pte_addr = pd_deref(g, pte_offset, V12PD_TARGET(pde.alt_target));
			if (IS_ERR_OR_NULL(pte_addr)) {
				printk(KERN_ERR "[nvdebug] %s: Unable to resolve %#lx in GPU %s to a kernel-accessible address. Error %ld.\n", __func__, pte_offset, pd_target_to_text(V12PD_TARGET(pde.alt_target)), -PTR_ERR(pte_addr));
				return 0;
			}
			// Read page table entry, avoiding readq
			pte.raw = readl(pte_addr + 4);
			pte.raw <<= 32;
			pte.raw |= readl(pte_addr);
			// Skip non-present PTEs
			if (!pte.is_present)
				continue;
			printk_debug(KERN_DEBUG "[nvdebug] PTE for phy addr %#018llx, ap '%s', vol '%d', priv '%d', ro '%d', no_atomics '%d' (raw: %#018llx)\n", ((u64)pte.addr) << 12, target_to_text(pte.target), pte.is_volatile, pte.is_privileged, pte.is_readonly, pte.atomics_disabled, pte.raw);
			// If we find a matching PTE, return its virtual address
			if ((uint64_t)pte.addr << 12 == addr_to_find && pte.target == addr_to_find_aperture)
				return i << NV_MMU_PT_V1_LSB[0] | j << NV_MMU_PT_V1_LSB[1];
		}
	} while (++i < NV_MMU_PT_V1_SZ[0]);
	return 0;
}

/* GPU Virtual address -> Physical address ("forward" translation) for V1 tables
  (See `translate_page_directory()` for documentation.)
*/
int translate_v1_page_directory(struct nvdebug_state *g,
                                page_dir_config_t pd_config,
                                uint64_t addr_to_find,
                                uint64_t *found_addr /* out */,
                                enum INST_TARGET *found_aperture /* out */) {
	page_dir_entry_v1_t pde;
	page_tbl_entry_v1_t pte;
	uintptr_t pde_idx, pde_phys, pte_idx, pte_phys;
	void __iomem *pte_kva, *pde_kva;

	*found_addr = 0;
	*found_aperture = TARGET_INVALID;

	// Make sure that the query is page-aligned (likely mistake otherwise)
	if (addr_to_find & 0xfff) {
		printk(KERN_WARNING "[nvdebug] Attempting to translate unaligned address %#llx in translate_v1_page_directory()!\n", addr_to_find);
		return -EINVAL;
	}

	// This function only understands the Page Table Version 1 format
	if (pd_config.is_ver2) {
		printk(KERN_ERR "[nvdebug] Passed a Version 2 page table at %#018llx to translate_v1_page_directory()!\n", (uint64_t)pd_config.page_dir << 12);
		return -EINVAL;
	}

	// We only understand the Version 1 format when 128 KiB huge pages are in-use
	if (pd_config.is_64k_big_page) {
		printk(KERN_ERR "[nvdebug] Page Table Version 1 with 64 KiB huge pages is unsupported!\n");
		return -EINVAL;
	}

	printk_info(KERN_INFO "[nvdebug] Translating addr %#018llx in V1 page table with base %#018llx\n", (uint64_t)addr_to_find, (uint64_t)pd_config.page_dir << 12);

	// Shift bits which define PDE index to start at bit 0, and mask other bits
	pde_idx = (addr_to_find >> NV_MMU_PT_V1_LSB[0]) & (NV_MMU_PT_V1_SZ[0] - 1);
	// Compute VID_MEM/SYS_MEM address of page directory entry
	pde_phys = ((uint64_t)pd_config.page_dir << 12) + pde_idx * sizeof(page_dir_entry_v1_t);
	// Convert VID_MEM/SYS_MEM address to Kernel-accessible Virtual Address (KVA)
	pde_kva = pd_deref(g, pde_phys, INST2PD_TARGET(pd_config.target));
	if (IS_ERR_OR_NULL(pde_kva)) {
		printk(KERN_ERR "[nvdebug] %s: Unable to resolve %#lx in GPU %s to a kernel-accessible address. Error %ld.\n", __func__, pde_phys, target_to_text(pd_config.target), PTR_ERR(pde_kva));
		return PTR_ERR(pde_kva);
	}
	// Read page directory entry (readq seems to work fine; tested on GM204)
	pde.raw = readq(pde_kva);
	// Verify this PDE points to an array of page table entries
	if (pde.target == PD_TARGET_INVALID && pde.alt_target == PD_TARGET_INVALID)
		return -ENXIO;
	// TODO: Check for and handle huge pages
	printk_debug(KERN_DEBUG "[nvdebug] Found %s PDE pointing to PTEs @ %llx in ap '%d' (raw: %llx)\n", pde.alt_is_volatile ? "volatile" : "non-volatile", ((u64)pde.alt_addr) << 12, pde.alt_target, pde.raw);

	// Shift bits which define PTE index to start at bit 0, and mask other bits
	pte_idx = (addr_to_find >> NV_MMU_PT_V1_LSB[1]) & (NV_MMU_PT_V1_SZ[1] - 1);
	// Compute VID_MEM/SYS_MEM address of page table entry
	pte_phys = ((uint64_t)pde.alt_addr << 12) + pte_idx * sizeof(page_tbl_entry_v1_t);
	// Convert VID_MEM/SYS_MEM address to Kernel-accessible Virtual Address (KVA)
	pte_kva = pd_deref(g, pte_phys, V12PD_TARGET(pde.alt_target));
	if (IS_ERR_OR_NULL(pde_kva)) {
		printk(KERN_ERR "[nvdebug] %s: Unable to resolve %#lx in GPU %s to a kernel-accessible address. Error %ld.\n", __func__, pte_phys, pd_target_to_text(V12PD_TARGET(pde.alt_target)), PTR_ERR(pte_kva));
		return PTR_ERR(pte_kva);
	}
	// Read page table entry
	pte.raw = readq(pte_kva);
	// XXX: The above readq() is bogus on gk104 (returns -1). Potential issue of pd_deref's move of PRAMIN racing with the driver?
	if (!pte.is_present)
		return -ENXIO;
	printk_debug(KERN_DEBUG "[nvdebug] PTE for phy addr %#018llx, ap '%s', vol '%d', priv '%d', ro '%d', no_atomics '%d' (raw: %#018llx)\n", ((u64)pte.addr) << 12, target_to_text(pte.target), pte.is_volatile, pte.is_privileged, pte.is_readonly, pte.atomics_disabled, pte.raw);
	// Access PTE and return physical address
	*found_addr = (uint64_t)pte.addr << 12;
	*found_aperture = pte.target;
	return 0;
}

/* *** UNTESTED ***
// This is only relevant on pre-Kepler GPUs; not a current priority
#define NV_MMU_PT_V0_SZ 2048
#define NV_MMU_PT_V0_LSB 29
uint64_t search_v0_page_directory(struct nvdebug_state *g,
				  void __iomem *pde_offset,
				  void __iomem *(*off2addr)(struct nvdebug_state*, uint32_t),
				  uint32_t addr_to_find) {
	int j, i = 0;
	page_dir_entry_v0_t pde;
	page_tbl_entry_v0_t pte;
	void __iomem *pte_offset;
	// For each PDE
	do {
		// readq doesn't seem to work on BAR0
		pde.raw = readl(pde_offset + i * sizeof(page_dir_entry_v0_t) + 4);
		pde.raw <<= 32;
		pde.raw |= readl(pde_offset + i * sizeof(page_dir_entry_v0_t));
		//if (pde.raw)
		//printk(KERN_INFO "[nvdebug] Read raw PDE @ %x: %llx\n", pde_offset + i * sizeof(page_dir_entry_v1_t), pde.raw);
		// Skip unpopulated PDEs
		if (pde.type == NOT_PRESENT)
			continue;
		//printk(KERN_INFO "[nvdebug] PDE to %llx present\n", ((uint64_t)pde.addr) << 12);
		pte_offset = off2addr(g, ((uint64_t)pde.addr) << 12);
		// For each PTE
		for (j = 0; j < V0_PDE_SIZE2NUM[pde.sublevel_size]; j++) {
			pte.raw = readl(pte_offset + j * sizeof(page_tbl_entry_v0_t) + 4);
			pte.raw <<= 32;
			pte.raw |= readl(pte_offset + j * sizeof(page_tbl_entry_v0_t));
			// Skip non-present PTEs
			if (!pte.is_present)
				continue;
			// If we find a matching PTE, return its virtual address
			//if (pte.addr != 0x5555555)
			//	printk(KERN_INFO "[nvdebug] PTE for phy addr %llx %s\n", ((uint64_t)pte.addr) << 12, pte.is_present ? "present" : "non-present");
			if (pte.addr << 12 == addr_to_find)
				return i << NV_MMU_PT_V0_LSB | j << 12;
		}
	} while (++i < NV_MMU_PT_V0_SZ);
	return 0;  // No match
}
*/
