# NVIDIA GPU Preemption

MVP: Preempt current work on the GPU on the Jetson Xavier

Summary of approach: Create new runlist that excludes the current work and point the GPU to it

1. Obtain current runlist
2. Copy runlist to new location, skipping TSG of target to preempt
3. Write new runlist address to NV_PFIFO_RUNLIST, which will preempt current work

It's unclear if this approach is lower-overhead than that of Capodieci et al.
See approach Alternate 1 which is our new priority.

Notes:
- Each TSG (timeslice group) corresponds to one context (?)
- Runlist base must be 4k aligned
- nvgpu driver gets gk20a struct via container_of an inode which is a struct nvgpu_os_linux
- gk20a_writel is nvgpu_writel. Define is: `void nvgpu_writel(struct gk20a *g, u32 reg_addr, u32 value);`
- gk20a_readl is nvgpu_readl. Define is: `u32 nvgpu_readl(struct gk20a *g, u32 reg_addr);`

## Other approaches:

### Alternate 1:
    "2. Disable all channels in the containing TSG by writing ENABLE_CLR to TRUE
        in their channel RAM entries in NV_PCCSR_CHANNEL (see dev_fifo.ref).
     3. Initiate a preempt of the TSG via NV_PFIFO_PREEMPT or
        NV_PFIFO_RUNLIST_PREEMPT." (PBDMA, "Recovery procedure")

### Alternate 2:
 "3. Initiate a preempt of the engine by writing the bit associated with its
     runlist to NV_PFIFO_RUNLIST_PREEMPT.  This allows us to begin the preempt
     process prior to doing the slow register reads needed to determine whether
     the context has hit any interrupts or is hung.  Do not poll
     NV_PFIFO_RUNLIST_PREEMPT for the preempt to complete." (FIFO, "Context TSG tear-down procedure")

See `nvdebug.c` and `nvdebug.h` for implementation details.
