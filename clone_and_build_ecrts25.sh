#!/bin/bash
# Copyright 2025 Joshua Bakita
# Set up the environment for re-running the evaluation for Bakita and Anderson (ECRTS 2025)
# Does not effect anything outside of the working directory *except* that it loads the nvdebug module (it will prompt for sudo for this)
# Prerequisites: CUDA SDK at /usr/local/cuda + make + gcc + jq + kernel headers
# (`sudo apt install build-essential linux-headers-generic jq` on Debian/Ubuntu)

echo -e "\e[4m\e[1m***** Getting and Building libsmctrl *****\e[0m"
git clone http://rtsrv.cs.unc.edu/cgit/cgit.cgi/libsmctrl.git/ -b ecrts25-ae
cd libsmctrl
make all
cd ..

echo -e "\e[4m\e[1m***** Getting and Building nvdebug *****\e[0m"
git clone http://rtsrv.cs.unc.edu/cgit/cgit.cgi/nvdebug.git/ -b ecrts25-ae
cd nvdebug
make
cd ..

echo -e "\e[4m\e[1m***** Getting and Building cuda_scheduling_examiner *****\e[0m"
git clone https://github.com/JoshuaJB/cuda_scheduling_examiner_mirror.git -b ecrts25-ae
cd cuda_scheduling_examiner_mirror
sed -i "s/#LIBSMCTRL/LIBSMCTRL/" Makefile
make all -j8
cd ..

echo -e "\e[4m\e[1m***** Getting and Building gpu-microbench *****\e[0m"
git clone http://rtsrv.cs.unc.edu/cgit/cgit.cgi/gpu-microbench.git/
cd gpu-microbench
make all
cd ..

cd nvdebug
echo -e "\e[4m\e[1m***** Loading nvdebug *****\e[0m"
echo -e "\e[4m\e[1m***** Type password to authorize loading kernel module *****\e[0m"
sudo insmod nvdebug.ko
cd ..

echo -e "\e[4m\e[1m***** Complete *****\e[0m"
echo -e "\e[4m\e[1mRemember to update the variables at the top of evaluate_ecrts25.sh to match your GPU before continuing!\e[0m"
