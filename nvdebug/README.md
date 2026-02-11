# nvdebug
Copyright 2021-2024 Joshua Bakita

Written to support my research on increasing the throughput and predictability of NVIDIA GPUs when running multiple tasks.

Please cite the following papers if using this in any published work:

1. J. Bakita and J. Anderson, “Hardware Compute Partitioning on NVIDIA GPUs”, Proceedings of the 29th IEEE Real-Time and Embedded Technology and Applications Symposium (RTAS), pp. 54–66, May 2023.
2. J. Bakita and J. Anderson, “Demystifying NVIDIA GPU Internals to Enable Reliable GPU Management”, Proceedings of the 30th IEEE Real-Time and Embedded Technology and Applications Symposium (RTAS), pp. 294-305, May 2024.

## API Overview
This module creates a virtual folder at `/proc/gpuX` for each GPU X in the system.
The contained virtual files and folders, when read, return plaintext representations of various aspects of GPU state.
Some files can also be written to, to modify GPU behavior.

The major API surfaces are composed of the following files:

- Device info (`device_info`, `copy_topology`...)
- Scheduling examination (`runlist`)
- Scheduling manipulation (`enable/disable_channel`, `switch_to/preempt_tsg`, `resubmit_runlist`)

As of Sept 2024, this module supports all generations of NVIDIA GPUs.
Very old GPUs (pre-Kepler) are mostly untested, and only a few APIs are likely to work.
APIs are designed to detect and handle errors, and should not crash your system in any circumstance.

We now detail how to use each of these APIs.
The following documentation assumes you have already `cd`ed to `/proc/gpuX` for your GPU of interest.

## Device Information APIs

### List of Engines
Use `cat device_info` to get a pretty-printed breakdown of which engines this GPU contains.
This information is pulled directly from the GPU topology registers, and should be very reliable.

### Copy Engines
**See our RTAS'24 paper for why this is important.**

Use `cat copy_topology` to get a pretty-printed mapping of how each configured logical copy engine is serviced.
They may be serviced by a physical copy engines, or configured to map onto another logical copy engine.
This is pulled directly from the GPU copy configuration registers, and should be very reliable.
See the RTAS'24 paper listed in the "Citing" section for details on why this is important.

Use `cat num_ces` to get the number of available copy engines (number of logical copy engines on Pascal+).

### Texture Processing Cluster (TPC)/Graphics Processing Cluster (GPC) Floorsweeping
**See our RTAS'23 paper for why this is important.**

Use `cat num_gpcs` to get the number of __on-chip__ GPCs.
Not all these GPCs will necessarially be enabled.

Use `cat gpc_mask` to get a bit mask of which GPCs are disabled.
A set bit indicates a disabled GPC.
Bit 0 corresponds to GPC 0, bit 1 to GPC 1, and so on, up to the total number of on-chip GPCs.
Bits greater than the number of on-chip GPCs should be ignored (it may appear than non-existent GPCs are "disabled").

Use `cat num_tpc_per_gpc` to get the number of __on-chip__ TPCs per GPC.
Not all these TPCs will necessarially be enabled in every GPC.

Use `cat gpcX_tpc_mask` to get a bit mask of which TPCs are disabled for GPC X.
A set bit indicates a disabled TPC.
This API is only available on enabled GPCs.
Bits greater than the number of on-chip TPCs per GPC should be ignored (it may appear than non-existent TPCs are "disabled").

Example usage: To get the number of on-chip SMs on Volta+ GPUs, multiply the return of `cat num_gpcs` with `cat num_tpc_per_gpc` and multiply by 2 (SMs per TPC).

## Scheduling Examination and Manipulation
**See our RTAS'24 paper for some uses of this.**

Some of these APIs operate within the scope of a runlist.
`runlistY` represents one of the `runlist0`, `runlist1`, `runlist2`, etc folders.

Use `cat runlistY/runlist` to view the contents and status of all channels in runlist Y.
**This is nvdebug's most substantial API.**
The runlist is composed of time-slice groups (TSGs, also called channel groups in nouveau) and channels.
Channels are indented in the output to indicate that they below to the preceeding TSG.

Use `echo Z > disable_channel` or `echo Z > runlistY/disable_channel` to disable channel with ID Z.

Use `echo Z > enable_channel` or `echo Z > runlistY/enable_channel` to enable channel with ID Z.

Use `echo Z > preempt_tsg` or `echo Z > runlistY/preempt_tsg` to trigger a preempt of TSG with ID Z.

Use `echo Z > runlistY/switch_to_tsg` to switch the GPU to run only the specified TSG with ID Z on runlist Y.

Use `echo Y > resubmit_runlist` to resubmit runlist Y (useful to prompt newer GPUs to pick up on re-enabled channels).

## Error Interpretation
First check the kernel log to see if in includes more information about the error.
The following conventions are used for certain error codes:

- EIO, "Input/Output Error," is returned when an operation fails due to a bad register read.
- (Other errors may not have a consistent conventional meaning; see the implementation.)

## General Codebase Structure
- `nvdebug.h` defines and describes all GPU data structures. This does not depend on any kernel-internal headers.
- `nvdebug_entry.h` contains module startup, device detection, initialization, and module teardown logic.
- `runlist.c`, `bus.c`, and `mmu.c` describe Linux-independent (as far as practicable) GPU data structure accessors.
- `*_procfs.c` define `/proc/gpuX/` interfaces for reading or writing to GPU data structures.
- `nvdebug_linux.c` contains Linux-specific accessors.

## Known Issues and Workarounds

- The runlist-printing API does not work when runlist management is delegated to the GPU System Processor (GSP) (most Turing+ datacenter GPUs).
  To workaround, enable the `FALLBACK_TO_PRAMIN` define in `runlist.c`, or reload the `nvidia` kernel module with the `NVreg_EnableGpuFirmware=0` parameter setting.
  (Eg. on A100: end all GPU-using processes, then `sudo rmmod nvidia_drm nvidia_modeset nvidia_uvm nvidia; sudo modprobe nvidia NVreg_EnableGpuFirmware=0`.)
