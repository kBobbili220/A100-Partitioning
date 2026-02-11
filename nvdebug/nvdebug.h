/* Copyright 2024 Joshua Bakita
 * SPDX-License-Identifier: MIT
 *
 * File outline:
 * - Configuration options
 * - Runlist, preemption, and channel control (FIFO)
 * - Basic GPU information (MC)
 * - Detailed GPU information (PTOP, FUSE, and CE)
 * - PRAMIN, BAR1/2, and page table status
 * - Helper functions for nvdebug
 *
 * This function should not depend on any Linux-internal headers, and may be
 * included outside of nvdebug.
 *
 * Style: This file uses up to 82-character lines to accomodate 2-character
 * indented quotes from open-gpu-doc without reflowing.
 */
#include <linux/types.h>

// Fully defined in include/nvgpu/gk20a.h. We only pass around pointers to
// this, so declare as incomplete type to avoid pulling in the nvgpu headers.
struct gk20a;

// Uncomment to, upon BAR2 access failure, return a PRAMIN-based runlist pointer
// in get_runlist_iter(). In order for this pointer to remain valid, PRAMIN
// **must** not be moved during runlist traversal.
// - The Jetson TX2 has no BAR2, and stores the runlist in VID_MEM, so this
//   must be enabled to print the runlist on the TX2.
// - On the A100 in Google Cloud and H100 in Paperspace, as of Aug 2024, this is
//   needed, as nvdebug is not finding (at least) runlist0 mapped in BAR2/3.
// Automatically disables printing Instance Block and Context State while
// traversing the runlist, as these require conflicting uses of PRAMIN (it's
// needed to search the page tables for the Instance Block in BAR2/3, and to
// access anything in the Context State---aka CTXSW).
#define FALLBACK_TO_PRAMIN

// Starting offset for registers in the corresponding named range
// Programmable First-In First-Out unit; also known as "Host"
#define NV_PFIFO 0x00002000 // 8 KiB long; ends prior to 0x00004000
// Programmable Channel Control System RAM
#define NV_PCCSR 0x00800000 // 16 KiB long; ends prior to 0x00810000
// Programmable TOPology registers
#define NV_PTOP 0x00022400 // 1 KiB long; ends prior to 0x00022800

/* Runlist Channel
  A timeslice group (TSG) is composed of channels. Each channel is a FIFO queue
  of GPU commands. These commands are typically queued from userspace.

  Prior to Volta, channels could also exist independent of a TSG. These are
  called "bare channels" in the Jetson nvgpu driver.

  `INST_PTR` points to a GPU Instance Block which contains FIFO states, virtual
  address space configuration for this context, and a pointer to the page
  tables. All channels in a TSG point to the same GPU Instance Block (?).

  "RUNQUEUE_SELECTOR determines to which runqueue the channel belongs, and
  thereby which PBDMA will run the channel.  Increasing values select
  increasingly numbered PBDMA IDs serving the runlist.  If the selector value
  exceeds the number of PBDMAs on the runlist, the hardware will silently
  reassign the channel to run on the first PBDMA as though RUNQUEUE_SELECTOR had
  been set to 0.  (In current hardware, this is used by SCG on the graphics
  runlist only to determine which FE pipe should service a given channel.  A
  value of 0 targets the first FE pipe, which can process all FE driven engines:
  Graphics, Compute, Inline2Memory, and TwoD.  A value of 1 targets the second
  FE pipe, which can only process Compute work.  Note that GRCE work is allowed
  on either runqueue." (NVIDIA) Note that it appears runqueue 1 is the default
  for CUDA work on the Jetson Xavier.

  ENTRY_TYPE (T)        : type of this entry: ENTRY_TYPE_CHAN
  CHID (ID)             : identifier of the channel to run (overlays ENTRY_ID)
  RUNQUEUE_SELECTOR (Q) : selects which PBDMA should run this channel if
                          more than one PBDMA is supported by the runlist,
                          additionally, "A value of 0 targets the first FE
                          pipe, which can process all FE driven engines:
                          Graphics, Compute, Inline2Memory, and TwoD.  A value
                          of 1 targets the second FE pipe, which can only
                          process Compute work.  Note that GRCE work is allowed
                          on either runqueue.)"

  INST_PTR_LO           : lower 20 bits of the 4k-aligned instance block pointer
  INST_PTR_HI           : upper 32 bits of instance block pointer
  INST_TARGET (TGI)     : aperture of the instance block

  USERD_PTR_LO          : upper 24 bits of the low 32 bits, of the 512-byte-aligned USERD pointer
  USERD_PTR_HI          : upper 32 bits of USERD pointer
  USERD_TARGET (TGU)    : aperture of the USERD data structure

  Channels were around since at least Fermi, but were rearranged with Volta to
  add a USERD pointer, a longer INST pointer, and a runqueue selector flag.
*/
enum ENTRY_TYPE {ENTRY_TYPE_CHAN = 0, ENTRY_TYPE_TSG = 1};
enum INST_TARGET {TARGET_VID_MEM = 0, TARGET_INVALID = 1, TARGET_SYS_MEM_COHERENT = 2, TARGET_SYS_MEM_NONCOHERENT = 3};
static inline const char *target_to_text(enum INST_TARGET t) {
	switch (t) {
		case TARGET_VID_MEM:
			return "VID_MEM";
		case TARGET_SYS_MEM_COHERENT:
			return "SYS_MEM_COHERENT";
		case TARGET_SYS_MEM_NONCOHERENT:
			return "SYS_MEM_NONCOHERENT";
		default:
			return "INVALID";
	}
}

// Support: Volta, Ampere, Turing, Ampere, Hopper, Ada
struct gv100_runlist_chan {
// 0:63
	enum ENTRY_TYPE entry_type:1;
	uint32_t runqueue_selector:1;
	 uint32_t :2;
	enum INST_TARGET inst_target:2;
	 uint32_t :2;
	uint32_t userd_ptr_lo:24;
	uint32_t userd_ptr_hi:32;
// 64:128
	uint32_t chid:12;
	uint32_t inst_ptr_lo:20;
	uint32_t inst_ptr_hi:32;
} __attribute__((packed));

// Support: Fermi, Kepler*, Maxwell, Pascal
// *On Kepler (and older?), inst fields are unpopulated (ex. gk104)
struct gm107_runlist_chan {
	uint32_t chid:12;
	 uint32_t :1;
	enum ENTRY_TYPE entry_type:1;
	 uint32_t :18;
	uint32_t inst_ptr_lo:20;
	enum INST_TARGET inst_target:2;  // Totally guessing on this
	 uint32_t :10;
} __attribute__((packed));

#define gk110_runlist_chan gm107_runlist_chan

/* Runlist TSG (TimeSlice Group)
  The runlist is composed of timeslice groups (TSG). Each TSG corresponds
  to a single virtual address space on the GPU and contains `TSG_LENGTH`
  channels. These channels and virtual address space are accessible to the GPU
  host unit for use until the timeslice expires or a TSG switch is forcibly
  initiated via a write to `NV_PFIFO_PREEMPT`.

  timeslice = (TSG_TIMESLICE_TIMEOUT << TSG_TIMESLICE_SCALE) * 1024 nanoseconds

  ENTRY_TYPE (T)      : type of this entry: ENTRY_TYPE_TSG
  TIMESLICE_SCALE     : scale factor for the TSG's timeslice
  TIMESLICE_TIMEOUT   : timeout amount for the TSG's timeslice
  TSG_LENGTH          : number of channels that are part of this timeslice group
  TSGID               : identifier of the Timeslice group (overlays ENTRY_ID)

  TSGs appear to have been introduced with Kepler and stayed the same until
  they were rearranged at the time of channel rearrangement to support longer
  GPU instance addresses with Volta.

  According to nvgpu, "timeslice is measured with PTIMER [which may be] lower
  than 1GHz."
*/

// Support: Volta, Turing*, Ampere*, Hopper, Ada
// *These treat bits 4:11 (8 bits) as GFID (unused)
struct gv100_runlist_tsg {
// 0:63
	enum ENTRY_TYPE entry_type:1;
	 uint64_t :15;
	uint32_t timeslice_scale:4;
	 uint64_t :4;
	uint32_t timeslice_timeout:8;
	uint32_t tsg_length:8;
	 uint32_t :24;
// 64:128
	uint32_t tsgid:12;
	 uint64_t :52;
} __attribute__((packed));
#define MAX_TSGID (1 << 12)

// Support: Kepler (v2?), Maxwell, Pascal
// Same fields as Volta except tsg_length is 6 bits rather than 8
// Last 32 bits appear to contain an undocumented inst ptr
struct gk110_runlist_tsg {
	uint32_t tsgid:12;
	 uint32_t :1;
	enum ENTRY_TYPE entry_type:1;
	uint32_t timeslice_scale:4;
	uint32_t timeslice_timeout:8;
	uint32_t tsg_length:6;
	 uint32_t :32;
} __attribute__((packed));


enum PREEMPT_TYPE {PREEMPT_TYPE_CHANNEL = 0, PREEMPT_TYPE_TSG = 1};

/* Preempt a TSG or Channel by ID
  ID/CHID     : Id of TSG or channel to preempt
  IS_PENDING  : Is a context switch pending? (read-only)
  TYPE        : PREEMPT_TYPE_CHANNEL or PREEMPT_TYPE_TSG

  Support: Fermi*, Kepler, Maxwell, Pascal, Volta, Turing
  *Fermi only supports PREEMPT_TYPE_CHANNEL.
*/
#define NV_PFIFO_PREEMPT 0x00002634
typedef union {
	struct {
		uint32_t id:12;
		 uint32_t :8;
		bool is_pending:1;
		 uint32_t :3;
		enum PREEMPT_TYPE type:2;
		 uint32_t :6;
	} __attribute__((packed));
	uint32_t raw;
} pfifo_preempt_t;

/* Preempt a TSG or Runlist by ID
  Similar as on older GPUs (see above), but located at an offset in Runlist RAM.
  This means that there's one instance of this register for each runlist.

  IS_PENDING is now IS_TSG_PREEMPT_PENDING and IS_RUNLIST_PREEMPT_PENDING was
  added in the following bit (bit 22). As these fields are unused in nvdebug,
  we use the old structure for simplicity.

  TYPE is now better described as IS_TSG_PREEMPT. TYPE == 0 requests a preempt
  of the runlist (rather than a channel preemption, as on older GPUs).

  Support: Ampere, Hopper, Ada, [newer untested]
*/
#define NV_RUNLIST_PREEMPT_GA100 0x098
#define PREEMPT_TYPE_RUNLIST PREEMPT_TYPE_CHANNEL

/*
  "Initiate a preempt of the engine by writing the bit associated with its
  runlist to NV_PFIFO_RUNLIST_PREEMPT...  Do not poll NV_PFIFO_RUNLIST_PREEMPT
  for the preempt to complete." (open-gpu-doc)

  Useful for preempting multiple runlists at once.

  Appears to trigger an interrupt or some other side-effect on the Jetson
  Xavier, as the built-in nvgpu driver seems to be disturbed by writing to this.

  To select the runlist dynamically, use the BIT(nr) kernel macro.
  Example:
    runlist_preempt_t rl_preempt;
    rl_preempt.raw = nvdebug_readl(g, NV_PFIFO_RUNLIST_PREEMPT);
    rl_preempt.raw |= BIT(nr);
    nvdebug_writel(g, NV_PFIFO_RUNLIST_PREEMPT, rl_preempt.raw);

  Support: Maxwell, Pascal, Volta, Turing

  This register was deleted starting with Ampere, with functionality subsumed by
  the NV_RUNLIST_PREEMPT register.
*/
#define NV_PFIFO_RUNLIST_PREEMPT 0x00002638
typedef union {
	struct {
		bool runlist_0:1;
		bool runlist_1:1;
		bool runlist_2:1;
		bool runlist_3:1;
		bool runlist_4:1;
		bool runlist_5:1;
		bool runlist_6:1;
		bool runlist_7:1;
		bool runlist_8:1;
		bool runlist_9:1;
		bool runlist_10:1;
		bool runlist_11:1;
		bool runlist_12:1;
		bool runlist_13:1;
		 uint32_t :18;
	} __attribute__((packed));
	uint32_t raw;
} runlist_preempt_t;

/* Additional information on preempting from NVIDIA's driver (commit b1d0d8ece)
 * "From h/w team
 * Engine save can be blocked by eng  stalling interrupts.
 * FIFO interrupts shouldn’t block an engine save from
 * finishing, but could block FIFO from reporting preempt done.
 * No immediate reason to reset the engine if FIFO interrupt is
 * pending.
 * The hub, priv_ring, and ltc interrupts could block context
 * switch (or memory), but doesn’t necessarily have to.
 * For Hub interrupts they just report access counters and page
 * faults. Neither of these necessarily block context switch
 * or preemption, but they could.
 * For example a page fault for graphics would prevent graphics
 * from saving out. An access counter interrupt is a
 * notification and has no effect.
 * SW should handle page faults though for preempt to complete.
 * PRI interrupt (due to a failed PRI transaction) will result
 * in ctxsw failure reported to HOST.
 * LTC interrupts are generally ECC related and if so,
 * certainly don’t block preemption/ctxsw but they could.
 * Bus interrupts shouldn’t have anything to do with preemption
 * state as they are part of the Host EXT pipe, though they may
 * exhibit a symptom that indicates that GPU is in a bad state.
 * To be completely fair, when an engine is preempting SW
 * really should just handle other interrupts as they come in.
 * It’s generally bad to just poll and wait on a preempt
 * to complete since there are many things in the GPU which may
 * cause a system to hang/stop responding."
 */

/* Runlist Metadata (up through Volta)
  "Software specifies the GPU contexts that hardware should "run" by writing a
  list of entries (known as a "runlist") to a 4k-aligned area of memory (beginning
  at NV_PFIFO_RUNLIST_BASE), and by notifying Host that a new list is available
  (by writing to NV_PFIFO_RUNLIST).

  Submission of a new runlist causes Host to expire the timeslice of all work
  scheduled by the previous runlist, allowing it to schedule the channels present
  in the new runlist once they are fetched. SW can check the status of the runlist
  by polling NV_PFIFO_ENG_RUNLIST_PENDING. (see dev_fifo.ref NV_PFIFO_RUNLIST for
  a full description of the runlist submit mechanism).

  Runlists can be stored in system memory or video memory (as specified by
  NV_PFIFO_RUNLIST_BASE_TARGET). If a runlist is stored in video memory, software
  will have to execute flush or read the last entry written before submitting the
  runlist to Host to guarantee coherency." (volta/dev_ram.ref.txt)

  We only document the *_PFIFO_ENG_RUNLIST_*(i) read-only registers here (where
  i is a runlist index). Runlists are configured via the seperate, writable
  *_PFIFO_RUNLIST_* register; see open-gpu-doc for more.

  LEN         : Number of entries in runlist
  IS_PENDING  : Is runlist committed?
  PTR         : Pointer to start of 4k-aligned runlist (upper 28 of 40 bits)
  TARGET      : Aperture of runlist (video or system memory)

  Support: Fermi*, Kepler, Maxwell, Pascal, Volta
  *Fermi may expose ENG_RUNLING_* 8 bytes earlier, starting at 0x227C?
*/
#define NV_PFIFO_RUNLIST_BASE_GF100 0x00002270 // Write-only
#define NV_PFIFO_ENG_RUNLIST_BASE_GF100(i) (0x00002280+(i)*8) // Read-only
typedef union {
	struct {
		// NV_PFIFO_ENG_RUNLIST_BASE_* fields
		uint32_t ptr:28;
		enum INST_TARGET target:2;
		 uint32_t :2;
		// NV_PFIFO_ENG_RUNLIST_* fields
		uint16_t len:16;
		 uint32_t :4;
		bool is_pending:1; // Read-only from NV_PFIFO_ENG_RUNLIST...
		 uint32_t :11;
	} __attribute__((packed));
	struct {
		// NV_PFIFO_RUNLIST_* fields that differ from NV_PFIFO_ENG_RUNLIST_*
		 uint64_t :52;
		uint32_t id:4; // Write-only to NV_PFIFO_RUNLIST...
		 uint32_t :8;
	} __attribute__((packed));
	uint64_t raw;
} eng_runlist_gf100_t;

/*
  Starting with Turing, the separate registers for reading and writing runlist
  configuration were dropped in favor of read/write indexed registers. As part
  of this, the layout was modified to allow for larger runlist pointers (upper
  52 of 64 bits).

  Support: Turing, Ampere*, Hopper*, Ada*
  *Only the register layout
*/
// Support: Turing
#define NV_PFIFO_RUNLIST_BASE_TU102(i) (0x00002B00+(i)*16) // Read/write
#define NV_PFIFO_RUNLIST_SUBMIT_TU102(i) (0x00002B08+(i)*16) // Read/write
// Derived absolute maximum number of runlists
#define MAX_RUNLISTS_TU102 80 // On Turing; another register is at 0x00003000
#define MAX_RUNLISTS_GF100 34 // On Volta-; another register is at 0x00002390
typedef union {
	struct {
		enum INST_TARGET target:2;
		 uint32_t :10;
		uint64_t ptr:28;
		 uint32_t :24;
	} __attribute__((packed));
	uint64_t raw;
} runlist_base_tu102_t;

/*
  LEN                   : Read/Write
  OFFSET                : Read/Write
  PREEMPTED_TSGID       : Read-only
  VALID_PREEMPTED_TSGID : Read-only
  IS_PENDING            : Read-only
  PREEMPTED_OFFSET      : Read-only
*/
typedef union {
	struct {
		uint16_t len:16;
		uint16_t offset:16;
		uint32_t preempted_tsgid:14;
		bool valid_preempted_tsgid:1;
		bool is_pending:1;
		uint32_t preempted_offset:16;
	} __attribute__((packed));
	uint64_t raw;
} runlist_submit_tu102_t;

enum CHANNEL_STATUS {
	CHANNEL_STATUS_IDLE = 0,
	CHANNEL_STATUS_PENDING = 1,
	CHANNEL_STATUS_PENDING_CTX_RELOAD = 2,
	CHANNEL_STATUS_PENDING_ACQUIRE = 3,
	CHANNEL_STATUS_PENDING_ACQ_CTX_RELOAD = 4,
	CHANNEL_STATUS_ON_PBDMA = 5,
	CHANNEL_STATUS_ON_PBDMA_AND_ENG = 6,
	CHANNEL_STATUS_ON_ENG = 7,
	CHANNEL_STATUS_ON_ENG_PENDING_ACQUIRE = 8,
	CHANNEL_STATUS_ON_ENG_PENDING = 9,
	CHANNEL_STATUS_ON_PBDMA_CTX_RELOAD = 10,
	CHANNEL_STATUS_ON_PBDMA_AND_ENG_CTX_RELOAD = 11,
	CHANNEL_STATUS_ON_ENG_CTX_RELOAD = 12,
	CHANNEL_STATUS_ON_ENG_PENDING_CTX_RELOAD = 13,
	CHANNEL_STATUS_ON_ENG_PENDING_ACQ_CTX_RELOAD = 14,
};

/* RunList RAM (RLRAM)
  Starting with Ampere, the PFIFO register region no longer exists, and each
  engine has seperate runlist RAM and channel RAM. The register (BAR0) offset for
  Runlist RAM for each engine must be pulled from the runlist_pri_base field
  (RUNLIST Private Register BASE address) provided by PTOP.

 See get_runlist_ram() in runlist.c

 Support: Ampere+
*/
#define NV_RUNLIST_BASE_GA100 0x080
#define NV_RUNLIST_SUBMIT_GA100 0x088
#define NV_RUNLIST_CHANNEL_CONFIG_GA100 0x004

/* Channel RAM configuration, as contained in Runlist RAM

  NUM_CHANNELS_LOG2 : 1 << NUM_CHANNELS_LOG2 is the number of channel_ctrl_ga100_t
                      entries in the described Channel RAM region.
  BAR0_OFFSET       : BAR0_OFFSET << 4 is the register offset (off BAR0) for the
                      Channel RAM region.

  Support: Ampere+
*/
typedef union {
	struct {
		uint8_t num_channels_log2:4;
		uint32_t bar0_offset:28;
	}__attribute__((packed));
	uint32_t raw;
} runlist_channel_config_t;

/* Context Switch Timeout Configuration
  After a task's budget expires, there's a configurable grace period, a
  "timeout", within which the context needs to complete. After this timeout
  expires, an interrupt is raised to terminate the task.

  This register configures if such a timeout is enabled and how long the
  timeout is (the "period").

  Support: Volta, Turing
*/
#define NV_PFIFO_ENG_CTXSW_TIMEOUT 0x00002A0C
// Support: Ampere
#define NV_RUNLIST_ENGINE_CTXSW_TIMEOUT_CONFIG(i) (0x220+(i)*64)
typedef union {
	struct {
		uint32_t period:31;
		bool enabled:1;
	} __attribute__((packed));
	uint32_t raw;
} ctxsw_timeout_t;

/* Programmable Channel Control System RAM (PCCSR)
  512-entry array of channel control and status data structures.

  === Read/Write Fields ===
  INST_PTR             : Top 28 of 40 bits of page-aligned channel instance block.
                         Instance Block = (uint64_t)inst_ptr << 12.
  INST_TARGET          : Aperture of INST_PTR.
  INST_BIND            : Is the channel instance bound?
  NEXT                 : Is this the next channel to be scheduled in the runlist?

  === Read-Only Fields ===
  ENABLE               : Is this channel enabled? (Disabled channels are skipped
                         over by the runlist scheduler.)
  PBDMA_FAULTED^       : [UNKNOWN]
  ENG_FAULTED^         : [UNKNOWN]
  STATUS               : Status of this channel in regards to hardware. See enum
                         CHANNEL_STATUS.
  BUSY                 : [UNKNOWN]
  ^Field can be reset with a non-zero write.

  === Write-Only Fields ===
  FORCE_CTX_RELOAD     : [UNKNOWN]
  ENABLE_SET           : Enables the channel upon non-zero write.
  ENABLE_CLEAR         : Disables the channel upon non-zero write.
  FORCE_PBDMA_FAULTED* : [UNKNOWN]
  FORCE_ENG_FAULTED*   : [UNKNOWN]
  *Field only available on Turing.

  Support: Fermi, Maxwell, Pascal, Volta, Turing
  See also: manuals/turing/tu104/dev_fifo.ref.txt in NVIDIA's open-gpu-doc
*/
#define NV_PCCSR_CHANNEL_INST(i) (0x00800000+(i)*8)
// Maximum valid channel index in the PCCSR region
// Channel IDs start at 0, and there are 4096 bytes of 8-byte CCSR entries (per
// NV_PCCSR_CHANNEL_INST__SIZE_1 in at least Volta and Turing), yielding a total
// of 512 channel IDs, with a maximum ID of 511.
#define MAX_CHID 511
typedef union {
	struct {
// 0:31
		uint32_t inst_ptr:28;
		enum INST_TARGET inst_target:2;
		 uint32_t :1;
		bool inst_bind:1;
// 32:63
		bool enable:1;
		bool next:1;
		 uint32_t :6;
		bool force_ctx_reload:1;
		 uint32_t :1;
		bool enable_set:1;
		bool enable_clear:1;
		 uint32_t :8;
		bool force_pbdma_faulted:1;
		bool force_eng_faulted:1;
		bool pbdma_faulted:1;
		bool eng_faulted:1;
		enum CHANNEL_STATUS status:4;
		bool busy:1;
		 uint32_t :3;
	} __attribute__((packed));
	struct {
		uint32_t word1;
		uint32_t word2;
	} __attribute__((packed));
	uint64_t raw;
} channel_ctrl_gf100_t;

// TODO: Remove use of deprecated type name
typedef channel_ctrl_gf100_t channel_ctrl_t;

/* CHannel RAM (CHRAM) (PCCSR replacement on Ampere+)
  Starting with Ampere, channel IDs are no longer unique indexes into the
  global channel RAM region (PCCSR), but are indexes into per-runlist channel
  RAMs.

  As Channel RAM entries are now subsidiary to a runlist, they do not contain
  duplicate information, such as the instance pointer (to "result in smaller
  hardware" per ga100/dev_ram.ref.txt in open-gpu-doc).

  The new format retains and adds to the status information available about a
  channel, but does so via bit flags rather than an enum. Some bit flags are
  writable to trigger behavior previously dedicated to a bit (eg. writing to
  `ctx_reload` triggers the same behavior as writing to `force_ctx_reload` did).

  When the first bit (`is_write_one_clears_bits`) is set in this structure,
  writing a 1 to any field will clear, rather than set, it. Writing a 0 to any
  field is a no-op.

  All fields read/write, except the following are read-only: BUSY, ON_PBDMA,
  ON_ENG, PBDMA_BUSY, ENG_BUSY.

  Support: Ampere, Hopper, Ada (and newer likely)
  See also: manuals/ampere/ga100/dev_runlist.ref.txt in NVIDIA's open-gpu-doc
*/
typedef union {
	struct {
		bool is_write_one_clears_bits:1; // new
		bool enable:1;
		bool next:1;
		bool busy:1;
		bool pbdma_faulted:1; // write to force_pbdma_faulted
		bool eng_faulted:1; // write to force_eng_faulted
		bool on_pbdma:1; // breakout
		bool on_eng:1; // breakout
		bool pending:1; // breakout
		bool ctx_reload:1; // breakout; write to force_ctx_reload
		bool pbdma_busy:1; // breakout
		bool eng_busy:1; // new
		bool acquire_fail:1; // breakout
		 uint32_t :19;
	} __attribute__((packed));
	uint32_t raw;
} channel_ctrl_ga100_t;

/* Control word for runlist enable/disable.

  RUNLIST_N           : Is runlist n disabled? (1 == disabled, 0 == enabled)

  To select the runlist dynamically, use the BIT(nr) kernel macro.
  Disabling example:
    runlist_disable_t rl_disable;
    rl_disable.raw = nvdebug_readl(g, NV_PFIFO_SCHED_DISABLE);
    rl_disable.raw |= BIT(nr);
    nvdebug_writel(g, NV_PFIFO_SCHED_DISABLE, rl_disable.raw);
  Enabling example:
    runlist_disable_t rl_disable;
    rl_disable.raw = nvdebug_readl(g, NV_PFIFO_SCHED_DISABLE);
    rl_disable.raw &= ~BIT(nr);
    nvdebug_writel(g, NV_PFIFO_SCHED_DISABLE, rl_disable.raw);

  Support: Fermi, Kepler, Maxwell, Pascal, Volta, Turing
*/
#define NV_PFIFO_SCHED_DISABLE 0x00002630
// Support: Ampere
#define NV_RUNLIST_SCHED_DISABLE 0x094
typedef union {
	struct {
		bool runlist_0:1;
		bool runlist_1:1;
		bool runlist_2:1;
		bool runlist_3:1;
		bool runlist_4:1;
		bool runlist_5:1;
		bool runlist_6:1;
		bool runlist_7:1;
		bool runlist_8:1;
		bool runlist_9:1;
		bool runlist_10:1;
		 uint32_t :21;
	} __attribute__((packed));
	uint32_t raw;
} runlist_disable_t;

/* Read GPU descriptors from the Master Controller (MC)

  MINOR_REVISION  : Legacy (only used with Celvin in Nouveau)
  MAJOR_REVISION  : Legacy (only used with Celvin in Nouveau)
  IMPLEMENTATION  : Which implementation of the GPU architecture
  ARCHITECTURE    : Which GPU architecture

  CHIP_ID = IMPLEMENTATION + ARCHITECTURE << 4
  CHIP_ID         : Unique ID of all chips since Kelvin

  Support: Kelvin, Rankline, Curie, Tesla, Fermi, Kepler, Maxwell, Pascal,
           Volta, Turing, Ampere
*/
#define NV_MC_BOOT_0 0x00000000
#define NV_CHIP_ID_GP106 0x136 // Discrete GeForce GTX 1060
#define NV_CHIP_ID_GV11B 0x15B // Jetson Xavier embedded GPU

#define NV_CHIP_ID_FERMI 0x0C0
#define NV_CHIP_ID_KEPLER 0x0E0
#define NV_CHIP_ID_MAXWELL 0x120
#define NV_CHIP_ID_PASCAL 0x130
#define NV_CHIP_ID_VOLTA 0x140
#define NV_CHIP_ID_VOLTA_INTEGRATED 0x150
#define NV_CHIP_ID_TURING 0x160
#define NV_CHIP_ID_AMPERE 0x170
#define NV_CHIP_ID_HOPPER 0x180
#define NV_CHIP_ID_ADA 0x190
#define NV_CHIP_ID_BLACKWELL 0x1A0

inline static const char* ARCH2NAME(uint32_t arch) {
	switch (arch) {
	case 0x01:
		return "Celsius";
	case 0x02:
		return "Kelvin";
	case 0x03:
		return "Rankline";
	case 0x04:
	case 0x06: // 0x06 is (nForce 6XX integrated only)
		return "Curie";
	// 0x07 is unused/skipped
	case 0x05: // First Tesla card was released before the nForce 6XX
	case 0x08:
	case 0x09:
	case 0x0A:
		return "Tesla";
	// 0x0B is unused/skipped
	case 0x0C:
	case 0x0D:
		return "Fermi";
	case 0x0E:
	case 0x0F:
	case 0x11:
		return "Kepler";
	case 0x12:
		return "Maxwell";
	case 0x13:
		return "Pascal";
	case 0x14:
	case 0x15: // Volta integrated
		return "Volta";
	case 0x16:
		return "Turing";
	case 0x17:
		return "Ampere";
	case 0x18: // Despite the Chip ID, Hopper functionally proceeds Ada
		return "Hopper";
	case 0x19:
		return "Ada Lovelace";
	case 0x1A:
		return "Blackwell";
	case 0x1B:
		return "Rubin (?)";
	case 0x1F: // NVIDIA-internal simulator
		return "AMODEL";
	default:
		if (arch < 0x1A)
			return "[unknown historical architecture]";
		else
			return "[future]";
	}
}

typedef union {
	// Fields as defined in the NVIDIA reference
	struct {
		uint32_t minor_revision:4;
		uint32_t major_revision:4;
		 uint32_t reserved:4;
		 uint32_t :8;
		uint32_t implementation:4;
		uint32_t architecture:5;
		 uint32_t :3;
	} __attribute__((packed));
	uint32_t raw;
	// Arch << 4 + impl is also often used
	struct {
		 uint32_t :20;
		uint32_t chip_id:9;
		 uint32_t :3;
	} __attribute__((packed));
} mc_boot_0_t;

/* GPU engine information and control register offsets (GPU TOPology)
  Each engine is described by one or more entries (terminated by an entry with
  the `has_next_entry` flag unset) in the fixed-size PTOP_DEVICE_INFO table. A
  typical device, such as the graphics/compute engine and any copy engines, are
  described by three entries, one of each type.

  The PTOP_DEVICE_INFO table is sparsely populated (entries of type
  INFO_TYPE_NOT_VALID may be intermingled with valid entries), so any traversal
  code should check all NV_PTOP_DEVICE_INFO__SIZE_1 entries and not terminate
  upon reaching the first entry of INFO_TYPE_NOT_VALID.

  The fields for the Ampere version of the GPU are a strict subset of those for
  the earlier versions.

  INFO_TYPE          : Is this a DATA, ENUM, or ENGINE_TYPE table entry?
  HAS_NEXT_ENTRY     : Does the following entry refer to the same engine?

  == INFO_TYPE_DATA fields ==
  PRI_BASE           : BAR0 base = (PRI_BASE << 12) aka 4k aligned.
  INST_ID            : "Note that some instanced [engines] (such as logical copy
                       engines aka LCE) share a PRI_BASE across all [engines] of
                       the same engine type; such [engines] require an additional
                       offset: instanced base = BAR0 base + stride * INST_ID.
  FAULT_ID_IS_VALID  : Does this engine have its own bind point and fault ID
                       with the MMU?
  FAULT_ID           : "The MMU fault id used by this [engine]. These IDs
                       correspond to the NV_PFAULT_MMU_ENG_ID define list."

  == INFO_TYPE_ENUM fields ==
  ENGINE_IS_VALID    : Is this engine a host engine?
  ENGINE_ENUM        : "[T]he host engine ID for the current [engine] if it is
                       a host engine, meaning Host can send methods to the
                       engine. This id is used to index into any register array
                       whose __SIZE_1 is equal to NV_HOST_NUM_ENGINES.  A given
                       ENGINE_ENUM can be present for at most one device in the
                       table.  Devices corresponding to all ENGINE_ENUM ids 0
                       through NV_HOST_NUM_ENGINES - 1 must be present in the
                       device info table."
  RUNLIST_IS_VALID   : Is this engine a host engine with a runlist?
  RUNLIST_ENUM       : "[T]he Host runlist ID on which methods for the current
                       [engine] should be submitted... The runlist id is used to
                       index into any register array whose __SIZE_1 is equal to
                       NV_HOST_NUM_RUNLISTS. [Engines] corresponding to all
                       RUNLIST_ENUM ids 0 through NV_HOST_NUM_RUNLISTS - 1 must
                       be present in the device info table."
  INTR_IS_VALID      : Does this device have an interrupt?
  INTR_ENUM          : Interrupt ID for use with "the NV_PMC_INTR_*_DEVICE
                       register bitfields."
  RESET_IS_VALID     : Does this engine have a reset ID?
  RESET_ENUM         : Reset ID for use indexing the "NV_PMC_ENABLE_DEVICE(i)
                       and NV_PMC_ELPG_ENABLE_DEVICE(i) register bitfields."

  == INFO_TYPE_ENGINE_TYPE fields ==
  ENGINE_TYPE        : What type of engine is this? (see ENGINE_TYPES_NAMES) 

  Support: Kepler, Maxwell, Pascal, Volta, Turing, Ampere
  See also: manuals/volta/gv100/dev_top.ref.txt in open-gpu-doc.
*/

#define NV_PTOP_DEVICE_INFO_GK104(i) (0x00022700+(i)*4)
#define NV_PTOP_DEVICE_INFO__SIZE_1_GK104 64
enum DEVICE_INFO_TYPE {INFO_TYPE_NOT_VALID = 0, INFO_TYPE_DATA = 1, INFO_TYPE_ENUM = 2, INFO_TYPE_ENGINE_TYPE = 3};
enum ENGINE_TYPES {
	ENGINE_GRAPHICS = 0, // GRAPHICS [/compute]
	ENGINE_COPY0 = 1, // [raw/physical] COPY #0
	ENGINE_COPY1 = 2, // [raw/physical] COPY #1
	ENGINE_COPY2 = 3, // [raw/physical] COPY #2

	ENGINE_MSPDEC = 8, // Picture DECoder
	ENGINE_MSPPP = 9, // [Video] Picture Post Processor
	ENGINE_MSVLD = 10, // [Video] Variable Length Decoder
	ENGINE_MSENC = 11, // [Video] ENCoding
	ENGINE_VIC = 12, // Video Image Compositor
	ENGINE_SEC = 13, // SEquenCer [?]
	ENGINE_NVENC = 14, // Nvidia Video ENCoder
	ENGINE_NVENC1 = 15, // Nvidia Video ENCoder #1
	ENGINE_NVDEC = 16, // Nvidia Video DECoder

	ENGINE_IOCTRL = 18, // I/O ConTRoLler [of NVLINK at least]
	ENGINE_LCE = 19, // Logical Copy Engine
	ENGINE_GSP = 20, // Gpu System Processor (Volta+)
	ENGINE_NVJPG = 21, // NVidia JPeG [Decoder] (Turing+)
	ENGINE_OFA = 22, // Optical Flow Accelerator (Turing+)
	ENGINE_FLA = 23, // [NVLink] Fabric Logical Addressing (Ampere+)
	// The HSHUB existed before Hopper, but was not considered an engine
	ENGINE_HSHUB = 24, // High Speed HUB (Hopper+)
	// C2C is the link for the TH500 (Grace) CPU/GPU combo
	ENGINE_C2C = 25, // Chip 2 Chip [Link] (Hopper+) (?)
	ENGINE_FSP = 26, // Firmware (?) Secure Processor (Hopper+)
};
#define ENGINE_TYPES_LEN 27
static const char* const ENGINE_TYPES_NAMES[ENGINE_TYPES_LEN] = {
	"Graphics/Compute",
	"COPY0",
	"COPY1",
	"COPY2",
	"Unknown Engine ID#4",
	"Unknown Engine ID#5",
	"Unknown Engine ID#6",
	"Unknown Engine ID#7",
	"MSPDEC: Picture Decoder",
	"MSPPP: Post Processing",
	"MSVLD: Variable Length Decoder",
	"MSENC: Encoder",
	"VIC: Video Image Compositor",
	"SEC: Sequencer",
	"NVENC: NVIDIA Video Encoder",
	"NVENC1: NVIDIA Video Encoder #1",
	"NVDEC: NVIDIA Video Decoder",
	"Unknown Engine ID#17",
	"IOCTRL: I/O Controller",
	"LCE: Logical Copy Engine",
	"GSP: GPU System Processor",
	"NVJPG: NVIDIA JPEG Decoder",
	"OFA: Optical Flow Accelerator",
	"FLA: Fabric Logical Addressing",
	"HSHUB: High Speed Hub",
	"C2C",
	"FSP",
};

typedef union {
	// DATA type fields
	struct {
		enum DEVICE_INFO_TYPE info_type:2;
		bool fault_id_is_valid:1;
		uint32_t fault_id:7;
		 uint32_t :2;
		uint32_t pri_base:12;
		 uint32_t :2;
		uint32_t inst_id:4;
		uint32_t is_not_enum2:1;
		bool has_next_entry:1;
	} __attribute__((packed));
	// ENUM type fields
	struct {
		 uint32_t :2;
		bool reset_is_valid:1;
		bool intr_is_valid:1;
		bool runlist_is_valid:1;
		bool engine_is_valid:1;
		 uint32_t :3;
		uint32_t reset_enum:5;
		 uint32_t :1;
		uint32_t intr_enum:5;
		 uint32_t :1;
		uint32_t runlist_enum:4;
		 uint32_t :1;
		uint32_t engine_enum:4;
		 uint32_t :2;
	} __attribute__((packed));
	// ENGINE_TYPE type fields
	struct {
		 uint32_t :2;
		enum ENGINE_TYPES engine_type:29;
		 uint32_t :1;
	} __attribute__((packed));
	uint32_t raw;
} ptop_device_info_gk104_t;

/* GPU TOPology on Ampere and newer GPUs
  On Ampere+, the array of device topology entries continues to describe all GPU
  engines, but the layout is entirely different to principly accomodate a
  pointer to the runlist configuration region for each engine. (Runlist
  configuration was moved out of the Host (PFIFO) region into per-engine spaces
  starting with Ampere.)

  Parsing is somewhat more difficult than with the older version, as entries
  no longer include an `info_type`. Instead, each entry has 1--3 subrows, where
  `has_next_entry` is 0 for the last subrow.

  Empty rows should be skipped.

  HAS_NEXT_ENTRY   : Is the following entry a descriptor of the same engine?

  == Subrow 1 fields ==
  FAULT_ID         : [UNKNOWN]
  INST_ID          : [UNKNOWN]
  ENGINE_TYPE      : Enumerated name of the type of engine. (Seemingly identical
                     to ENGINE_ENUM in old PTOP layout.)

  == Subrow 2 fields ==
  RESET_ID         : [UNKNOWN]
  PRI_BASE         : [UNKNOWN]
  IS_ENGINE        : Does this entry describe an engine with a runlist? (Seemingly
                     identical to RUNLIST_IS_VALID in old PTOP layout.)

  == Subrow 3 fields ==
  RUNLIST_PRI_BASE : Offset in BAR0 of the RunList RAM (RLRAM) region for the
                     runlist of this engine.
  RLENG_ID         : What is the per-runlist ID of this engine?

  Support: Ampere, Hopper, Ada (and newer likely)
  See also: hw_top_ga100.h in nvgpu (NVIDIA's open-source Jetson GPU driver)
*/
#define NV_PTOP_DEVICE_INFO_GA100(i) (0x00022800+(i)*4)
#define NV_PTOP_DEVICE_INFO_LEN_GA100 0x000224fc
#define NV_PTOP_DEVICE_INFO_LEN_SHIFT_GA100 20
#define NV_PTOP_DEVICE_INFO__SIZE_1_GA100(g) (nvdebug_readl(g, NV_PTOP_DEVICE_INFO_LEN_GA100) >> NV_PTOP_DEVICE_INFO_LEN_SHIFT_GA100)

typedef union {
	// _info type fields
	struct {
		uint32_t fault_id:11;
		 uint32_t :5;
		uint32_t inst_id:8;
		enum ENGINE_TYPES engine_type:7; // "type_enum"
		bool has_next_entry:1;
	} __attribute__((packed));
	// _info2 type fields
	struct {
		uint32_t reset_id:8;
		uint32_t pri_base:18; // "device_pri_base"
		 uint32_t :4;
		uint32_t is_engine:1;
		 uint32_t :1;
	} __attribute__((packed));
	struct {
		uint32_t rleng_id:2;
		 uint32_t :8;
		uint32_t runlist_pri_base:16;
		 uint32_t :6;
	} __attribute__((packed));
	uint32_t raw;
} ptop_device_info_ga100_t;

/* Graphics Processing Cluster (GPC) on-chip information
  The GPU's Compute/Graphics engine is subdivided into Graphics Processing
  Clusters (also known as GPU Processing Clusters, starting with Ampere).

  Each GPC is subdivided into Texture Processing Clusters (TPCs) which contain
  Streaming Multiprocessors (SMs).

  The number of these units etched onto the chip may vary from the number
  enabled and software-visible. These registers expose the number of on-chip
  GPCs, the number of on-chip TPCs inside a GPC.

  Support: Fermi through (at least) Blackwell
*/
#define NV_PTOP_SCAL_NUM_GPCS 0x00022430
#define NV_PTOP_SCAL_NUM_TPC_PER_GPC 0x00022434

/* Graphics Processing Cluster (GPC) enablement information
  (See above for a description of GPCs and TPCs.)

  The number of on-chip GPCs and TPCs enabled is driven by:
  1) Manufacturing errors which make some units nonfunctional.
  2) Commercialization decisions about how many units should be enabled for a
     specific GPU model.

  Generally, reason (1) drives disablement early in product manufacturing,
  whereas, as the manufacturing process matures, (2) steps in to ensure
  consistency between early-manufactured and late-manufactured products.

  On-chip fuses are used to dictate which units are enabled and disabled. These
  registers expose the fuse configuration for GPCs, and the TPCs in each GPC.

  FUSE_GPC            : Bitmask of which GPCs are enabled
  FUSE_TPC_FOR_GPC(i) : Bitmask of which TPCs are enabled for GPC i

  Support: Maxwell through Blackwell
           Note the registers were relocated starting with Ampere.
*/
#define NV_FUSE_GPC_GM107 0x00021c1c
#define NV_FUSE_TPC_FOR_GPC_GM107(i) (0x00021c38+(i)*4)
#define NV_FUSE_GPC_GA100 0x00820c1c
#define NV_FUSE_TPC_FOR_GPC_GA100(i) (0x00820c38+(i)*4)

/* Logical Copy Engine (LCE) Information
  Every GPU has some number of copy engines which can process transfers to,
  from, or within a GPU. Up until Maxwell, the hardware engines were directly
  accessible, and this register exposes how many there are.

  Starting with Pascal, an additional layer of indirection was added---logical
  copy engines. Only logical copy engines can be directly dispatched to, and
  there are normally more logical copy engines than there are physical ones. On
  Pascal+ this register stores the number of logical copy engines.

  SCAL_NUM_CES : Number of externally accessible copy engines

  Errata: Incorrectly reports "3" on Jetson TX1 and TX2. Should report "1" to be
          consistent with PTOP data.

  Support: Kepler through (at least) Blackwell
  Also see dev_ce.ref.txt of NVIDIA's open-gpu-doc for info.
*/
#define NV_PTOP_SCAL_NUM_CES 0x00022444
// Defined LCE->PCE mapping offset from nvgpu (same as ce_pce2lce_config_r(i) in nvgpu)
#define NV_LCE_FOR_PCE_GP100 0x0010402c
#define NV_LCE_FOR_PCE_GV100(i) (0x00104040+(i)*4)
#define NV_LCE_FOR_PCE_GA100(i) (0x00104100+(i)*4)
/* GRaphics Copy Engine (GRCE) Information
  "There's two types of CE... ASYNC_CEs which are copy engines with their own
  runlists and GRCEs which are CEs that share a runlist with GR." (nvgpu,
  ioctl_ctrl.c)

  Starting with Pascal, the GRCEs are LCEs 0 and 1, but have the added capability
  to share a PCE with another LCE. (Normally a PCE may only be associated with
  one LCE.) These registers include that configuration, which should only be set
  if no PCE has been directly associated with the specific GRCE.

  Support: Pascal through (at least) Ada
           Note that Volta through Ada use a different bit format than Pascal.
*/
// Defined max number of GRCEs for a GPU (TX2 has only one)
# define NV_GRCE_MAX 2
// Defined GRCE->CE mapping offsets from nvgpu
#define NV_GRCE_FOR_CE_GP100(i) (0x00104034+(i)*4)
#define NV_GRCE_FOR_CE_GA100(i) (0x001041c0+(i)*4)

// Struct for use with nvdebug_reg_range_read()
union reg_range {
	struct {
		uint32_t offset;
		uint8_t start_bit;
		uint8_t stop_bit;
	};
	uint64_t raw;
};

/* Physical Copy Engine (PCE) information
  On Pascal GPUs or newer, this register complements the above information by
  exposing which, and how many, physical copy engines are enabled on the GPU.

  CE_PCE_MAP : A bitmask, where a set bit indicates that the PCE for that index
               is enabled (not floorswept) on this GPU. Count the number of set
               bits to get the number of PCEs. Note that this may be bogus if
               the GPU has not been used since reset.

  Support: Pascal through (at least) Blackwell
  Also see dev_ce.ref.txt of NVIDIA's open-gpu-doc for info.
*/
#define NV_CE_PCE_MAP 0x00104028
#define NV_CE_PCE_MAP_SIZE 32


/* Location of the 1Kb instance block with page tables for the BAR1/2 regions.

  On the H100, the "BAR1 block" describes what is actually BAR2, and the
  "BAR2 block" describes BAR4.

  PTR : Upper 28 bits of the 40-bit, (4k-aligned) address where the instance
        block configuration is for the listed BAR region.

  "Hopper+ uses 64-bit BARs, so GPU BAR2 should be at BAR4/5 and GPU BAR1 is at
  BAR2/3" (open-gpu-kernel-modules)
*/
// Support: Fermi through Ampere, Ada
#define NV_PBUS_BAR1_BLOCK 0x00001704
#define NV_PBUS_BAR2_BLOCK 0x00001714
typedef union {
	struct {
		uint32_t ptr:28;
		enum INST_TARGET target:2;
		 uint32_t :1; // disable_cya_debug for BAR2
		bool is_virtual:1;
	} __attribute__((packed));
	uint32_t raw;
	struct {
		uint32_t map:30;
		 uint32_t :2;
	} __attribute__((packed));
} bar_config_block_t;

// Support: Hopper, Blackwell+
// This is a "VREG" (virtual register?) in the documentation, meaning that it
// needs the VREG base added first.
#define NV_VIRTUAL_FUNCTION_FULL_PHYS_OFFSET 0x00B80000
#define NV_VIRTUAL_FUNCTION_PRIV_FUNC_BAR2_BLOCK (NV_VIRTUAL_FUNCTION_FULL_PHYS_OFFSET+0x00000F70)
typedef union {
	struct {
		bool is_pending:1;
		bool is_outstanding:1;
		 uint32_t :7;
		bool is_virtual:1;
		enum INST_TARGET target:2;
		uint64_t ptr:40;
		 uint32_t :12;
	} __attribute__((packed));
	uint64_t raw;
	struct {
		 uint32_t :10;
		uint32_t map:22;
		 uint32_t :32;
	} __attribute__((packed));
} bar_config_block_gh100_t;

/* BAR0 PRAMIN (Private RAM Instance) window configuration
  One of the oldest ways to access video memory on NVIDIA GPUs is by using
  a configurable 1MB window into VRAM which is mapped into BAR0 (register)
  space starting at offset NV_PRAMIN. This is still supported on NVIDIA GPUs
  and appear to be used today to bootstrap page table configuration.

  Why is it mapped at a location called NVIDIA Private RAM Instance? Because
  this used to point to the entirety of intance RAM, which was seperate from
  VRAM on older NVIDIA GPUs.

  BASE    : Base of window >> 16 in [TARGET] virtual address space
  TARGET  : Which address space BASE points into

  Note: This seems to be set to 0x0bff00000 - 0x0c0000000 at least sometimes
*/
// Support: Tesla 2.0, Fermi, Kepler, Maxwell, Pascal, Turing, Ampere, Ada
#define NV_PBUS_BAR0_WINDOW 0x00001700
// On Hopper, and Blackwell+, TARGET must always be 0 (VIDMEM)
// Support: Hopper, Blackwell+
#define NV_XAL_EP_BAR0_WINDOW_BASE 0x0010fd40
typedef union {
	struct {
		uint32_t base:24;
		enum INST_TARGET target:2;
		 uint32_t :6;
	} __attribute__((packed));
	uint32_t raw;
} bar0_window_t;

// Support: Tesla 2.0 through (at least) Blackwell
#define NV_PRAMIN 0x00700000  // Goes until 0x00800000 (1MB window)
#define NV_PRAMIN_LEN 0x00100000

/* Page Directory Base (PDB) configuration for an instance block

  Note: "Volta only supports [the] new page table format [V2] and [a] 64KB big
        page size" (kern_gmmu_gv100.c in open-gpu-kernel-modules).
  Support: Tesla 2.0* through Ampere, Ada
   *FAULT_REPLAY_* fields are Pascal+ only
  See also: dev_ram.h (open-gpu-kernel-modules) or dev_ram.ref.txt (open-gpu-doc)

  It appears that on Hopper, IS_VER2 continues to mean IS_VER2, but if unset, the
  alternative is VER3.
*/
#define NV_PRAMIN_PDB_CONFIG_OFF 0x200
typedef union {
	struct {
		enum INST_TARGET target:2;
		uint32_t is_volatile:1;
		 uint32_t :1;
		bool fault_replay_tex:1;
		bool fault_replay_gcc:1;
		 uint32_t :4;
		bool is_ver2:1; // XXX: Not on Hopper. May be set or not for same page_dir.
		bool is_64k_big_page:1;  // 128Kb otherwise
		uint32_t page_dir_lo:20;
		uint32_t page_dir_hi:32;
	} __attribute__((packed));
	struct {
		 uint32_t :12;
		uint64_t page_dir:52;
	} __attribute__((packed));
	uint64_t raw;
} page_dir_config_t;

/* NVIDIA GMMU (GPU Memory Management Unit) uses page tables that are mostly
  straight-forward starting with Pascal ("page table version 2"), except for a
  few quirks (like 16-byte PDE0 entries, but all other entries are 8 bytes).

  All you really need to know is that any given Page Directory Entry (PDE)
  contains a pointer to the start of a 4k page densely filled with PDEs or Page
  Table Entries (PTEs).

  == Page Table Refresher ==
  Page tables convert virtual addresses to physical addresses, and they do this
  via a tree structure. Leafs (PTEs) contain a physical address, and the path
  from root to leaf is defined by the virtual address. Non-leaf nodes are PDEs.
  When decending, the virtual address is sliced into pieces, and one slice is
  used at each level (as an index) to select the next-visited node (in level+1).

  V2 of NVIDIA's page table format uses 4 levels of PDEs and a final level of
  PTEs. How the virtual address is sliced to yield an index into each level and
  a page offset is shown by Fig 1.

  == Figure 1 ==
  Page Offset (12 bits) <---------------------------------------+
  Page Table Entry (PTE) (9 bits) <--------------------+        |
  Page Directory Entry (PDE) 0 (8 bits) <-----+        |        |
  PDE1 (9 bits) <--------------------+        |        |        |
  PDE2 (9 bits) <-----------+        |        |        |        |
  PDE3 (2 bits) <--+        |        |        |        |        |
                   ^        ^        ^        ^        ^        ^
  Virtual addr: [48, 47] [46, 38] [37, 29] [28, 21] [20, 12] [11, 0]

  The following arrays merely represent different projections of Fig. 1, and
  only one is strictly needed to reconstruct all the others. However, due to
  the complexity of page tables, we include all of these to aid in readability.

  Support: Pascal, Volta, Turing, Ampere, Hopper*, Ada, Blackwell*
  Note: *Hopper introduces Version 3 Page Tables, but is backwards-compatible.
         The newer version adds a PD4 level to support 57-bit virtual
         addresses, and slightly shifts the PDE and PTE fields.

  See also: gp100-mmu-format.pdf in open-gpu-doc. In open-gpu-kernel-modules
            this is synonymously the "NEW" and "VER2" layout.
*/
// How many nodes/entries per level in V2 of NVIDIA's page table format
static const int NV_MMU_PT_V2_SZ[5] = {4, 512, 512, 256, 512};
// Size in bytes of an entry at a particular level
static const int NV_MMU_PT_V2_ENTRY_SZ[5] = {8, 8, 8, 16, 8};
// Which bit index is the least significant in indexing each page level
static const int NV_MMU_PT_V2_LSB[5] = {47, 38, 29, 21, 12};

// Important: Aperture keys are different with PDEs
enum PD_TARGET {
	PD_AND_TARGET_INVALID = 0,  // b000
	PD_AND_TARGET_VID_MEM = 2,  // b010
	PD_AND_TARGET_SYS_MEM_COHERENT = 4,  // b100
	PD_AND_TARGET_SYS_MEM_NONCOHERENT = 6,  // b110
	PTE_AND_TARGET_VID_MEM = 1,  // b001
	PTE_AND_TARGET_PEER = 3,  // b011
	PTE_AND_TARGET_SYS_MEM_COHERENT = 5,  // b101
	PTE_AND_TARGET_SYS_MEM_NONCOHERENT = 7,  // b111
};
// The low bit is unset on page directory (PD) targets
#define IS_PD_TARGET(target) (!(target & 0x1u))
// Convert from an enum INST_TARGET to an enum PD_TARGET
#define INST2PD_TARGET(target) ((target & 0x2) ? (target << 1) : (!target) << 1)
// Convert from an enum V1_PD_TARGET to an enum PD_TARGET
#define V12PD_TARGET(target) (target << 1)
static inline const char *pd_target_to_text(enum PD_TARGET t) {
	switch (t) {
		case PD_AND_TARGET_INVALID:
			return "INVALID";
		case PD_AND_TARGET_VID_MEM:
		case PTE_AND_TARGET_VID_MEM:
			return "VID_MEM";
		case PTE_AND_TARGET_PEER:
			return "PEER";
		case PD_AND_TARGET_SYS_MEM_COHERENT:
		case PTE_AND_TARGET_SYS_MEM_COHERENT:
			return "SYS_MEM_COHERENT";
		case PD_AND_TARGET_SYS_MEM_NONCOHERENT:
		case PTE_AND_TARGET_SYS_MEM_NONCOHERENT:
			return "SYS_MEM_NONCOHERENT";
		default:
			return "UNKNOWN";
	}
}

/* Page Directory Entry/Page Table Entry V2 type
  We consider the least-significant bit to be 0, and use interval notation.
  Example: The first 8 bits of an address could be identically described as
  (8, 0], [7, 0], [7, -1), or (8, -1).

  ADDR   : Bits [35, 12] of the physical address; bits [11, 0] are 0. This is the
           full 36-bit address for VID_MEM or PEER targets. For SYS_MEM targets,
           use ADDR_W to include bits [57, 36] (per gp100-mmu-format.pdf, Pascal
           only uses bits [46, 0]---a 2^47 = 128 TiB physical address space).
           Points to first entry of the next level of the page table (in a PDE),
           or the start of the physical frame (in a PTE).
  ADDR_W : Bits [57, 12] of a SYS_MEM address. Only necessary for physical
           addresses over 128 TiB. See ADDR.
  IS_VOL : If set, the pointed-to frame should not be cached in the GPU L2 cache.
           This applies to PDEs (then the pointed-to page table/directory frame
           will not be cached), and to PTEs (then the pointed to data frame will
           not be cached). This **does not apply to VID_MEM**, except on Tegra.
  NO_ATS : "GPUs which support ATS [Volta+] perform a parallel lookup on both
           ATS and GMMU page tables. The ATS lookup can be disabled by setting a
           bit in the GMMU page tables. All GPUs which support ATS use the same
           mechanism (a bit in PDE1), and have the same PDE1 coverage (512MB)."
           (nvidia-uvm/uvm_mmu.h)
           Other parts of the nvidia-uvm documentation note that disabling the
           ATS lookup helps performance.

  Note: As the meaning of target (bits 2:1) at a PDE-level changes if the
        entry is a large-page PTE or not. To simply the logic, we combine them
        into a single target field to simplify comparisons.

  See also: gp100-mmu-format.pdf in open-gpu-doc.
*/
#define TARGET_PEER 1
typedef union {
	// Page Directory Entry (PDE)
	struct {
		enum PD_TARGET target:3;
		bool is_volatile:1;
		 uint32_t :1;
		bool no_ats:1; // Set to disable PCIe (?) Address Translation Services
		 uint32_t :2;
		uint32_t addr:24;
		 uint32_t __unused1;
	} __attribute__((packed));
	// Page Table Entry (PTE)
	struct {
		bool is_pte:1;
		enum INST_TARGET aperture:2;
		 uint32_t __is_volatile:1;
		bool is_encrypted:1;
		bool is_privileged:1;
		bool is_readonly:1;
		bool atomics_disabled:1;
		 uint32_t __addr:24;
		 uint32_t __unused2;
	} __attribute__((packed));
	// For wide addresses in PTEs or PDEs; only used if target is SYS_MEM
	struct {
		 uint32_t __overlap:8;
		uint64_t addr_w:46;
		 uint32_t __unused3:10;
	} __attribute__((packed));
	uint64_t raw_w;
} page_dir_entry_t;

/* GMMU Page Tables Version 1
  These page tables contain 2 levels and are used in the Fermi, Kepler, and
  Maxwell architectures to support a 40-bit virtual address space.

  Version 1 Page Tables may be configured to support either 64 KiB or 128 KiB
  large pages. Table addressing differs between the modes---even if the table
  contains no large pages. The format for 4 KiB pages in each mode is shown
  below.

  V1 of NVIDIA's page table format uses 1 level of PDEs and a level of PTEs.
  How the virtual address is sliced to yield an index into each level and a
  page offset is shown by Fig 1 and Fig 2 (for 64 KiB and 128 KiB large page
  modes respectively).

  == Figure 1: 64 KiB mode ==
  Page Offset (12 bits) <----------------------------------+
  Page Table Entry (PTE) (13 bits) <--------------+        |
  Page Directory Entry (PDE) (13 bits) <-+        |        |
                                         ^        ^        ^
                     Virtual address: [39, 26] [25, 12] [11, 0]

  == Figure 2: 128 KiB mode ==
  Page Offset (12 bits) <----------------------------------+
  Page Table Entry (PTE) (14 bits) <--------------+        |
  Page Directory Entry (PDE) (12 bits) <-+        |        |
                                         ^        ^        ^
                     Virtual address: [39, 27] [26, 12] [11, 0]


  Support: Fermi, Kepler, Maxwell, Pascal*
  Note: *Pascal introduces Version 2 Page Tables, but is backwards-compatible.
  Note: We only implement the 128-KiB-large-page mode in nvdebug.

  See also: mm_gk20a.c in nvgpu (Jetson GPU driver) and kern_gmmu_fmt_gm10x.c
            in open-gpu-kernel-modules (open-source NVRM variant). This is
            synonymously the "VER1" and unversioned layout in
            open-gpu-kernel-modules, with some differences noted in Appdx 1.

  == Appdx 1 ==
  In open-gpu-kernel-modules, the unversioned MMU layout adds:
  - Bit 35: NV_MMU_PTE_LOCK synonym for NV_MMU_PTE_ATOMIC_DISABLE
  - Bit 62: NV_MMU_PTE_READ_DISABLE overlapping NV_MMU_PTE_COMPTAGLINE
  - Bit 63: NV_MMU_PTE_WRITE_DISABLE overlapping NV_MMU_PTE_COMPTAGLINE
  And removes:
  - Bit 40, 41, 42, 43 from NV_MMU_PTE_KIND
  The PDE layouts are identical. Given that the unversioned defines seem to
  predate renaming and/or field extension/relocation, they are likely artifacts
  from the page table development process, and have no meaning now.
*/
// Number of entries in the PDE and PTE levels
static const int NV_MMU_PT_V1_SZ[2] = {4096, 16384}; // 2^12 and 2^14
// Which bit index is the least significant in indexing each page level
static const int NV_MMU_PT_V1_LSB[2] = {27, 12};

// V1 Page Directory Entry target
enum V1_PD_TARGET {
	PD_TARGET_INVALID = 0,
	PD_TARGET_VID_MEM = 1,
	PD_TARGET_SYS_MEM_COHERENT = 2,
	PD_TARGET_SYS_MEM_NONCOHERENT = 3,
};
// V1 Page Directory Entry (PDE)
typedef union {
// Large page fields
	struct {
// 0:32
		enum V1_PD_TARGET target:2;
		 uint32_t :2; // Documented as "PDE_SIZE"?
		uint64_t addr:28;  // May be wider?
// 32:63
		 uint32_t :3;
		uint32_t is_volatile:1; // Might have counted wrong?
		 uint32_t :28;
	} __attribute__((packed));
// Small page fields
	struct {
// 0:32
		 uint32_t :32;
// 32:63
		enum V1_PD_TARGET alt_target:2;
		uint32_t alt_is_volatile:1; // Might have counted wrong?
		 uint32_t :1;
		uint64_t alt_addr:28;
	} __attribute__((packed));
	uint64_t raw;
} page_dir_entry_v1_t;

// V1 Page Table Entry (PTE)
typedef union {
	struct {
// 0:32
		bool is_present:1;
		bool is_privileged:1;
		bool is_readonly:1;
		bool is_encrypted:1;
		uint64_t addr:28;
// 32:63
		bool is_volatile:1;
		enum INST_TARGET target:2;
		bool atomics_disabled:1;
		uint32_t kind:8;
		uint32_t comptag:20;
	} __attribute__((packed));
	uint64_t raw;
} page_tbl_entry_v1_t;

/* GMMU Page Tables Version 0
  This page table contains 2 levels to support a 40-bit virtual address space,
  and is used in the Tesla (2.0?) architecture.

  It is unclear what NVIDIA calls this page table layout. It predates V1, so we
  call it V0.

  See also: https://envytools.readthedocs.io/en/latest/hw/memory/g80-vm.html
 */
/*
// What size pages are in the pointed-to page table?
enum V0_PDE_TYPE {NOT_PRESENT = 0, PAGE_64K = 1, PAGE_16K = 2, PAGE_4K = 3};
// How large is the pointed-to page table?
enum V0_PDE_SIZE {PDE_SZ_128K = 0, PDE_SZ_32K = 1, PDE_SZ_16K = 2, PDE_SZ_8K = 3};
// Given a page table size, how many entries does it have?
static const int V0_PDE_SIZE2NUM[4] = {128*1024, 32*1024, 16*1024, 8*1024};

// PDE V0 (nv50/Tesla)
typedef union {
	struct {
		enum V0_PDE_TYPE type:2;
		enum INST_TARGET target:2;
		 uint32_t :1;
		enum V0_PDE_SIZE sublevel_size:2;
		 uint32_t :5;
		uint32_t addr:28; // Bits [12, 39] of the 40-bit page table address
		 uint32_t :24;
	} __attribute__((packed));
	uint64_t raw;
} page_dir_entry_v0_t;

// PTE V0 (nv50) for small pages
typedef union {
	struct {
		bool is_present:1;
		 uint32_t :2;
		bool is_readonly:1;
		enum INST_TARGET target:2;
		bool is_privileged:1;
		uint32_t contig_blk_sz:3;
		 uint32_t :2;
		uint32_t addr:28; // Bits [12, 39] of the 40-bit frame address
		uint32_t storage_type:7;  // ???
		uint32_t compression_mode:2;  // ???
		uint32_t compression_tag:12;  // ???
		bool is_long_partition_cycle:1;  // ???
		bool is_encrypted:1;
		 uint32_t :1;
	} __attribute__((packed));
	uint64_t raw;
} page_tbl_entry_v0_t;
*/

/* Fifo Context RAM (RAMFC) and channel INstance RAM (RAMIN)

  Each channel is configured with a 4 KiB instance block. The prefix of this
  block is referred to as RAMFC and stores channel-specific state for the Host
  (aka PFIFO).

  "A GPU instance block is a block of memory that contains the state
  for a GPU context.  A GPU context's instance block consists of Host state,
  pointers to each engine's state, and memory management state.  A GPU instance
  block also contains a pointer to a block of memory that contains that part of a
  GPU context's state that a user-level driver may access.  A GPU instance block
  fits within a single 4K-byte page of memory."

  "The NV_RAMFC part of a GPU-instance block contains Host's part of a virtual
  GPU's state.  Host is referred to as "FIFO". "FC" stands for FIFO Context.
  When Host switches from serving one GPU context to serving a second, Host saves
  state for the first GPU context to the first GPU context's RAMFC area, and loads
  state for the second GPU context from the second GPU context's RAMFC area."

  "Every Host word entry in RAMFC directly corresponds to a PRI-accessible
  register.  For a description of the contents of a RAMFC entry, please see the
  description of the corresponding register in "manuals/dev_pbdma.ref".  The
  offsets of the fields within each entry in RAMFC match those of the
  corresponding register in the associated PBDMA unit's PRI space."

  In summary, RAMFC includes details such as the head and tail of the pushbuffer,
  and RAMIN includes details such as the page table configuration(s).

  The instance-global page table (as defined in the PDB field) is only used for
  GPU engines which do not support subcontexts (non-VEID engines).

  **Not all documented fields are currently populated below.**

  Support: *Kepler, *Maxwell, *Pascal, Volta, Turing, Ampere, [newer untested]
  *Pre-Volta GPUs do not support subcontexts.
  See also: dev_ram.ref.txt and dev_pbdma.ref.txt in NVIDIA's open-gpu-doc
*/

// 16-byte (128-bit) substructure defining a subcontext configuration
typedef struct {
	page_dir_config_t pdb;
	uint32_t pasid:20; // Process Address Space ID (PASID) used for ATS
	 uint32_t :11;
	bool enable_ats:1; // Enable Address Translation Services (ATS)?
	 uint32_t pad;
} __attribute__((packed)) subcontext_ctrl_t;

typedef struct {
// Start RAMFC (512 bytes)
	 uint32_t pad[43];
	uint32_t fc_target:5; // NV_RAMFC_TARGET; off 43
	 uint32_t :27;
	 uint32_t pad2[17];
	uint32_t fc_config_l2:1; // NV_RAMFC_CONFIG; off 61
	 uint32_t :3;
	uint32_t fc_config_ce_split:1;
	uint32_t fc_config_ce_no_throttle:1;
	 uint32_t :2;
	uint32_t fc_config_is_priv:1; // ...AUTH_LEVEL
	 uint32_t :3;
	uint32_t fc_config_userd_writeback:1; // ...USERD_WRITEBACK
	 uint32_t :19;
	 uint32_t pad3[1];
	uint32_t fc_chan_info_scg:1; // ...SET_CHANNEL_INFO_SCG_TYPE
	 uint32_t :7;
	uint32_t fc_chan_info_veid:6; // ...SET_CHANNEL_INFO_VEID
	uint32_t fc_chan_info_chid:12; // ...SET_CHANNEL_INFO_CHID
	 uint32_t :6;
	 uint32_t pad4[64];
// End RAMFC
// Start RAMIN
	page_dir_config_t pdb;
	 uint32_t pad5[2];
	// WFI_TARGET appears to be ignored if WFI_IS_VIRTUAL
	uint32_t engine_wfi_target:2; // NV_RAMIN_ENGINE_WFI_TARGET; off 132
	uint32_t engine_wfi_is_virtual:1;
	 uint32_t :9;
	// WFI_PTR points to a CTXSW block (documented below)
	uint64_t engine_wfi_ptr:52; // NV_RAMIN_ENGINE_WFI_PTR_LO/_HI; off 132--133
	uint32_t engine_wfi_veid:6; // NV_RAMIN_ENGINE_WFI_VEID; off 134; VEID == Subcontext ID
	 uint32_t :26;
	uint32_t pasid:20; // NV_RAMIN_PASID; off 135; "Process Address Space ID"
	 uint32_t :11;
	bool enable_ats:1;
	 uint32_t pad6[30];
	uint64_t subcontext_pdb_valid; // NV_RAMIN_SC_PDB_VALID; off 166-167
	subcontext_ctrl_t subcontext[64]; // NV_RAMIN_SC_*; off 168-424
} __attribute__((packed)) instance_ctrl_t;

// Context types
enum CTXSW_TYPE {
	CTXSW_UNDEFINED = 0x0,
	CTXSW_OPENGL = 0x8,
	CTXSW_DX9 = 0x10,
	CTXSW_DX10 = 0x11,
	CTXSW_DX11 = 0x12,
	CTXSW_COMPUTE = 0x20,
	CTXSW_HEADER = 0x21 // A per-subcontext header
};
static inline const char *ctxsw_type_to_text(enum CTXSW_TYPE t) {
	switch (t) {
		case CTXSW_UNDEFINED:
			return "[None]";
		case CTXSW_OPENGL:
			return "OpenGL";
		case CTXSW_DX9:
		case CTXSW_DX10:
		case CTXSW_DX11:
			return "DirectX";
		case CTXSW_COMPUTE:
			return "Compute";
		case CTXSW_HEADER:
			return "Header";
		default:
			return "UNKNOWN";
	}
}

// Preemption modes:
// WFI: Wait For Idle (preempt on idle)
// CTA: Cooperative Thread Array-level Preemption (preempt at end of block)
// CILP: Compute-Instruction-Level Preemption (preempt at end of instruction)
enum GRAPHICS_PREEMPT_TYPE {PREEMPT_WFI, PREEMPT_GFXP};
enum COMPUTE_PREEMPT_TYPE {_PREEMPT_WFI, PREEMPT_CTA, PREEMPT_CILP};
static inline const char *compute_preempt_type_to_text(enum COMPUTE_PREEMPT_TYPE t) {
	switch (t) {
		case PREEMPT_WFI:
			return "WFI";
		case PREEMPT_CTA:
			return "CTA";
		case PREEMPT_CILP:
			return "CILP";
		default:
			return "INVALID";
	}
}
static inline const char *graphics_preempt_type_to_text(enum GRAPHICS_PREEMPT_TYPE t) {
	switch (t) {
		case PREEMPT_WFI:
			return "WFI";
		case PREEMPT_GFXP:
			return "GFXP";
		default:
			return "INVALID";
	}
}

/* ConTeXt SWitch control block (CTXSW)
  Support: Maxwell*, Pascal**, Volta, Turing, Ampere, Ada
   *Nothing except for CONTEXT_ID and TYPE
   **Except as noted
  See also: manuals/volta/gv100/dev_ctxsw.ref.txt in open-gpu-doc
            and hw_ctxsw_prog_*.h in nvgpu
*/
// (Note that this layout changes some generation-to-generation)
typedef struct context_switch_block {
	 uint32_t pad[3];
	enum CTXSW_TYPE type:6; // Unused except when type CTXSW_HEADER?
	 uint32_t :26;
	 uint32_t pad2[26];
	// The context buffer ptr fields are in an opposite-of-typical order, so we
	// can't merge them into a single context_buffer_ptr field.
	uint32_t context_buffer_ptr_hi; // Volta+ only
	uint32_t context_buffer_ptr_lo; // Volta+ only
	enum GRAPHICS_PREEMPT_TYPE graphics_preemption_options:32;
	enum COMPUTE_PREEMPT_TYPE compute_preemption_options:32;
	 uint32_t pad3[18];
	uint32_t num_wfi_save_operations;
	uint32_t num_cta_save_operations;
	uint32_t num_gfxp_save_operations;
	uint32_t num_cilp_save_operations;
	 uint32_t pad4[4];
	uint32_t context_id;
	// [There are more fields not yet added here.]
} __attribute__((packed)) context_switch_ctrl_t;

/* VRAM Information

  If ECC is disabled:
    bytes = (magnitude << scale) * 1024 * 1024
  If ECC is enabled:
    bytes = ((magnitude << scale) * 1024 * 1024) / 16 * 15

  Support: Pascal, Volta, Turing, [more?]
 */
#define NV_FB_MMU_LOCAL_MEMORY_RANGE 0x00100ce0
typedef union  {
	struct {
		uint32_t scale:4;
		uint32_t mag:6;
		uint32_t:20;
		bool is_ecc:1;
		uint32_t:1;
	} __attribute__((packed));
	uint32_t raw;
} memory_range_t;

static inline uint64_t memory_range_to_bytes(memory_range_t range) {
	// ECC takes a byte out of available memory for parity data
	if (range.is_ecc)
		return ((range.mag << range.scale) * 1024ull * 1024ull) / 16 * 15;
	else
		return (range.mag << range.scale) * 1024ull * 1024ull;
}

/* Begin nvdebug types and functions */

// __iomem is only defined when building as a kernel module, so conditionally
// define it to allow including this header outside the kernel.
#ifndef __iomem
#define __iomem
#endif

// Vendor ID for PCI devices manufactured by NVIDIA
#define NV_PCI_VENDOR 0x10de
struct nvdebug_state {
	// Pointer to the mapped base address of the GPU control registers (obtained
	// via ioremap() originally). For embedded GPUs, we extract this from their
	// struct nvgpu_os_linux. For discrete GPUs, we create our own mapping of
	// BAR0 with pci_iomap(). Access via nvgpu_readl/writel functions.
	void __iomem *regs;
	// Depending on the architecture, BAR2 or BAR3 are used to access PRAMIN
	union {
		void __iomem *bar2;
		void __iomem *bar3;
	};
	int chip_id;
	// Additional state from the built-in driver. Only set on Jetson boards
	struct gk20a *g;
	// Pointer to PCI device needed for pci_iounmap and pci_resource_start
	struct pci_dev *pcid;
	// Pointer to platform device needed for platform_get_resource
	struct platform_device *platd;
	// Pointer to generic device struct (both platform and pcie devices)
	struct device *dev;
#ifdef __KERNEL__
	// List used by mmu.c to track allocated pages for page directories/tables
	struct list_head pd_allocs;
#endif
};

// This disgusting macro is a crutch to work around the fact that runlists were
// different prior to Volta.
#define VERSIONED_RL_ACCESSOR(_ENTRY_TYPE, type, prop) \
	__attribute__((unused)) \
	static type (prop)(const struct nvdebug_state *g, const void *raw) { \
		if (g->chip_id >= NV_CHIP_ID_VOLTA) { \
			const struct gv100_runlist_ ## _ENTRY_TYPE *entry = (struct gv100_runlist_ ## _ENTRY_TYPE*)raw; \
			return entry->prop; \
		} else if (g->chip_id >= NV_CHIP_ID_KEPLER) { \
			const struct gk110_runlist_ ## _ENTRY_TYPE *entry = (struct gk110_runlist_ ## _ENTRY_TYPE*)raw; \
			return entry->prop; \
		} else { \
			return (type)0; \
		} \
	}

VERSIONED_RL_ACCESSOR(chan, uint32_t, chid);
VERSIONED_RL_ACCESSOR(chan, uint32_t, inst_ptr_lo);
VERSIONED_RL_ACCESSOR(chan, enum INST_TARGET, inst_target);
VERSIONED_RL_ACCESSOR(tsg, uint32_t, tsgid);
VERSIONED_RL_ACCESSOR(tsg, enum ENTRY_TYPE, entry_type);
VERSIONED_RL_ACCESSOR(tsg, uint32_t, timeslice_scale);
VERSIONED_RL_ACCESSOR(tsg, uint32_t, timeslice_timeout);
VERSIONED_RL_ACCESSOR(tsg, uint32_t, tsg_length);


#define NV_RL_ENTRY_SIZE(g) \
	 ((g)->chip_id >= NV_CHIP_ID_VOLTA ? sizeof(struct gv100_runlist_tsg) : sizeof(struct gk110_runlist_tsg))

// chan and tsg should be pointers
#define for_chan_in_tsg(g, chan, tsg) \
        for (chan = (typeof(chan))(((u8*)tsg) + NV_RL_ENTRY_SIZE(g)); \
             (u8*)chan < ((u8*)tsg) + (1 + tsg_length(g, tsg)) * NV_RL_ENTRY_SIZE(g); \
             chan = (typeof(chan))(((u8*)chan) + NV_RL_ENTRY_SIZE(g)))

#define next_tsg(g, tsg) \
        (typeof(tsg))((u8*)(tsg) + NV_RL_ENTRY_SIZE(g) * (tsg_length(g, tsg) + 1))

struct runlist_iter {
	// Pointer to either a TSG or channel entry (they're the same size)
	void *curr_entry;
	// This should be set to tsg_length + 1 when a TSG is reached, and
	// decremented each time _next() is called. This allows us to
	// track which channels are and are not part of the TSG.
	int entries_left_in_tsg;
	// Number of entries in runlist
	int len;
	// (Ampere+ only) Offset to the per-runlist "Runlist RAM" register region.
	// This includes the offset for Channel RAM (per-runlist on Ampere+).
	uint32_t runlist_pri_base;
};

#define NVDEBUG_MAX_DEVICES 8
extern struct nvdebug_state g_nvdebug_state[NVDEBUG_MAX_DEVICES];

// Defined in runlist.c
int get_runlist_ram(
	struct nvdebug_state *g,
	int rl_id,
	uint32_t *rl_ram_off /* out */);
int get_runlist_iter(
	struct nvdebug_state *g,
	int rl_id,
	struct runlist_iter *rl_iter /* out */);
int preempt_tsg(struct nvdebug_state *g, uint32_t rl_id, uint32_t tsg_id);
int preempt_runlist(struct nvdebug_state *g, uint32_t rl_id);
int resubmit_runlist(struct nvdebug_state *g, uint32_t rl_id, uint32_t off);
instance_ctrl_t *instance_deref(
	struct nvdebug_state *g,
	uint64_t instance_addr,
	enum INST_TARGET instance_target);
context_switch_ctrl_t *get_ctxsw(
	struct nvdebug_state *g,
	instance_ctrl_t *inst);
int set_channel_preemption_mode(
	struct nvdebug_state *g,
	uint32_t chan_id,
	uint32_t rl_id,
	enum COMPUTE_PREEMPT_TYPE mode);

// Defined in mmu.c
uint64_t search_page_directory(
	struct nvdebug_state *g,
	page_dir_config_t pd_config,
	uint64_t addr_to_find,
	enum INST_TARGET addr_to_find_aperture);
int translate_page_directory(
	struct nvdebug_state *g,
	page_dir_config_t pd_config,
	uint64_t addr_to_find,
	uint64_t *found_addr /* out */,
	enum INST_TARGET *found_aperture /* out */);
int map_page_directory(
	struct nvdebug_state *g,
	page_dir_config_t pd_config,
	uint64_t paddr_to_map,
	uint64_t vaddr_to_find,
	enum INST_TARGET paddr_target,
	bool huge_page);
int gc_page_directory(
	struct nvdebug_state *g,
	bool force);
uint64_t search_v1_page_directory(
	struct nvdebug_state *g,
	page_dir_config_t pd_config,
	uint64_t addr_to_find,
	enum INST_TARGET addr_to_find_aperture);
int translate_v1_page_directory(
	struct nvdebug_state *g,
	page_dir_config_t pd_config,
	uint64_t addr_to_find,
	uint64_t *found_addr /* out */,
	enum INST_TARGET *found_aperture /* out */);
// Defined in bus.c
int addr_to_pramin_mut(struct nvdebug_state *g, uint64_t addr, enum INST_TARGET target);
int get_bar2_pdb(struct nvdebug_state *g, page_dir_config_t* pd /* out */);

// Some portions of nvdebug can be included from kernel- or user-space (just
// this file at present). In order for these compiled object files to be
// usable in either setting, the appropriate version of the following functions
// must be selected at link-time. Unfortunately, this precludes inlining (as
// the implementation of an inline function must be known at compile time)
// Implementations of these functions are provided for kernel-space by
// nvdebug_linux.c.
uint32_t nvdebug_readl(struct nvdebug_state *s, uint32_t r);
uint64_t nvdebug_readq(struct nvdebug_state *s, uint32_t r);
void nvdebug_writel(struct nvdebug_state *s, uint32_t r, uint32_t v);
void nvdebug_writeq(struct nvdebug_state *s, uint32_t r, uint64_t v);
