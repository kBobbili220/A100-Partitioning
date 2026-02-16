#!/usr/bin/env python3
# Copyright 2025 Joshua Bakita
# This requires python3, python3-numpy, and python3-matplotlib and an X display.
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

PCT_A=57

# Setting plt.rcParams MUST be in a different cell than the imports, or it won't apply
plt.rcParams["figure.figsize"] = (16,6)
plt.rcParams['figure.dpi'] = 96 # DPI used by Inkscape (not actually necessary, but makes plots more readable here)

plt.rcParams["pdf.use14corefonts"] = "True" # So that it doesn't try to embed Tex Gyre Heros
plt.rcParams["font.sans-serif"] = ["TeX Gyre Heros", "Nimbus Sans", "Helvetica", "Arimo"]
plt.rcParams["font.size"] = 8 # Default IEEEtran footnote size; 10 pt is default

# If getting warnings about "findfont: Font...not found. Falling back...", run:
# import matplotlib.font_manager
# matplotlib.font_manager._rebuild()

orange = "#f4b400"
blue = "#0277bd"
green = "#0f9d58"

def stats(arr):
    print("%d samples"%(len(arr)))
    print("max: %.2f"%(np.max(arr)))
    print("99th: %.2f"%(np.percentile(arr, 99)))
    print("mean: %.2f"%(np.mean(arr)))
    print("min: %.2f"%(np.min(arr)))
    print("stdev: %.2f"%(np.std(arr)))
    print("variance: %.2f"%(np.var(arr)))

def plot_startup():
	# Plot startup overheads
	base = np.loadtxt("startup_oh_baseline.log")/(1000*1000)
	mps  = np.loadtxt("startup_oh_mps.log")/(1000*1000)
	libsmctrl_wrapper = np.loadtxt("startup_oh_libsmctrl-wrapper.log")/(1000*1000)
	libsmctrl = np.loadtxt("startup_oh_libsmctrl.log")/(1000*1000)
	nvtaskset_gpc = np.loadtxt("startup_oh_nvtaskset-gpc.log")/(1000*1000)
	nvtaskset = np.loadtxt("startup_oh_nvtaskset.log")/(1000*1000)
	mig  = np.loadtxt("startup_oh_mig.log")/(1000*1000)

	plt.figure(figsize=(3.0,2.4))
	plt.bar(np.arange(7), [max(base)/1000, max(mig)/1000, max(mps)/1000, max(libsmctrl_wrapper)/1000, max(libsmctrl)/1000, max(nvtaskset_gpc)/1000, max(nvtaskset)/1000], color=["0.7", green, orange, "0.5", blue, "0.5", "0.5"])
	plt.xticks(np.arange(7), ["None", "MiG", "MPS", "libsmctrl-wrapper", "libsmctrl", "nvtaskset-gpc", "nvtaskset"], rotation=40)
	plt.xlabel("Partitioning Mechanism")
	plt.ylabel("Maximum Startup Overhead (s)")
	plt.tight_layout()
	plt.show()

def plot_launch():
	# Plot launch overheads
	base = np.loadtxt("launch_oh_baseline.log")/(1000)
	mps  = np.loadtxt("launch_oh_mps.log")/(1000)
	libsmctrl = np.loadtxt("launch_oh_libsmctrl.log")/(1000)
	mig  = np.loadtxt("launch_oh_mig.log")/(1000)

	plt.figure(figsize=(1.75,2.4))
	data = (base, mig, mps, libsmctrl)
	plt.boxplot(data, labels=("None", "MiG", "MPS", "nvsplit"), showfliers=False, whis=(0, 99))
	plt.xticks(rotation=40)
	plt.xlabel("Partitioning Mechanism")
	plt.ylabel("Launch Overhead (µs)")
	plt.tight_layout()
	plt.show()

def plot_granularity():
	# Plot granularity
	has_mig = os.system("ls ./cuda_scheduling_examiner_mirror/results/ecrts25_mig*") == 0
	if has_mig:
		os.system("python3 ./cuda_scheduling_examiner_mirror/scripts/view_granularity.py -w 400 -v 200 ./cuda_scheduling_examiner_mirror/results/ecrts25_mig* ./cuda_scheduling_examiner_mirror/results/ecrts25_mps_* ./cuda_scheduling_examiner_mirror/results/ecrts25_libsmctrl_*")
	else:
		os.system("python3 ./cuda_scheduling_examiner_mirror/scripts/view_granularity.py -w 400 -v 200 ./cuda_scheduling_examiner_mirror/results/ecrts25_mps* ./cuda_scheduling_examiner_mirror/results/ecrts25_mps_* ./cuda_scheduling_examiner_mirror/results/ecrts25_libsmctrl_*")

def plot_enforcement():
	# Plot partition enforcement results
	import json
	def load_stripped_cse(file):
		with open(file) as fp:
			return np.array(json.load(fp))

	i_baseline = load_stripped_cse("ecrts25_isol_baseline_stripped.json")
	i_none = load_stripped_cse("ecrts25_isol_none_rw_stripped.json")
	i_libsmctrl = load_stripped_cse("ecrts25_isol_libsmctrl_rw_stripped.json")
	i_mps = load_stripped_cse("ecrts25_isol_mps_rw_stripped.json")
	i_mig = load_stripped_cse("ecrts25_isol_mig_rw_stripped.json")

	plt.figure(figsize=(2.75,2.4)) # One-half LIPIcs

	# absolute minimum to 100th percentile
	data = (i_baseline*1/(PCT_A/100), i_none, i_mig, i_mps, i_libsmctrl)

	# "patch_artist" triggers drawing the boxes as closed boxes, rather than line segments.
	# This allows us to subsequently fill them with white, such that they overlap the grid lines.
	bplot = plt.boxplot(data, labels=("Scaled", "None", "MiG", "MPS", "nvsplit"), showfliers=False, whis=(0, 100), patch_artist=True)
	for patch in bplot['boxes']: patch.set_facecolor("white")

	#plt.xticks(rotation=40)
	plt.xlabel("Partitioning Mechanism")
	plt.ylabel("MM6144 Execution Time (ms)\n(versus 512MiB Random Walk)")
	plt.ylim(bottom=0)

	plt.gca().set_axisbelow(True)
	plt.grid()

	plt.tight_layout()
	plt.show()

	i_mb_none = load_stripped_cse("ecrts25_isol_none_mb_stripped.json")
	i_mb_libsmctrl = load_stripped_cse("ecrts25_isol_libsmctrl_mb_stripped.json")
	i_mb_mps = load_stripped_cse("ecrts25_isol_mps_mb_stripped.json")
	i_mb_mig = load_stripped_cse("ecrts25_isol_mig_mb_stripped.json")

	plt.figure(figsize=(2.75,2.4)) # One-half LIPIcs

	# absolute minimum to 100th percentile (max)
	data = (i_baseline*1/(PCT_A/100), i_mb_none, i_mb_mig, i_mb_mps, i_mb_libsmctrl)

	# "patch_artist" triggers drawing the boxes as closed boxes, rather than line segments.
	# This allows us to subsequently fill them with white, such that they overlap the grid lines.
	bplot = plt.boxplot(data, labels=("Scaled", "None", "MiG", "MPS", "nvsplit"), showfliers=False, whis=(0, 100), patch_artist=True)
	for patch in bplot['boxes']: patch.set_facecolor("white")

	#plt.xticks(rotation=40)
	plt.xlabel("Partitioning Mechanism")
	plt.ylabel("MM6144 Execution Time (ms)\n(vs. 10M-pixel 50k-iter. Mandelbrot)")
	# Add this to zoom
	#plt.ylim(bottom=250, top=260)

	plt.gca().set_axisbelow(True)
	plt.grid()

	plt.tight_layout()
	plt.show()

if len(sys.argv) > 1:
	if sys.argv[1] == "startup_oh":
		plot_startup()
	elif sys.argv[1] == "launch_oh":
		plot_launch()
	elif sys.argv[1] == "enforcement":
		plot_enforcement()
	elif sys.argv[1] == "granularity":
		plot_granularity()
	else:
		print("Usage: %s [startup_oh | launch_oh | enforcement | granularity]"%sys.argv[0])
		exit(1)
else:
	plot_startup()
	plot_launch()
	plot_enforcement()
	plot_granularity()
