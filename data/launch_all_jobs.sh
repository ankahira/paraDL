#!/usr/bin/env bash

# Data Parallelism jobs

# Alexnet

#qsub -g gaa50004 scripts/submit_alexnet_64_8.sh
#qsub -g gaa50004 scripts/submit_alexnet_128_16.sh
#qsub -g gaa50004 scripts/submit_alexnet_256_32.sh
#sleep 5

# New Experiments to test breaking point of data

qsub -g gaa50004 scripts/submit_alexnet_512_4.sh
qsub -g gaa50004 scripts/submit_alexnet_1024_4.sh
qsub -g gaa50004 scripts/submit_alexnet_2048_4.sh

#qsub -g gaa50004 scripts/submit_alexnet_2048_8.sh
#
#qsub -g gaa50004 scripts/submit_alexnet_4096_16.sh
#
#qsub -g gaa50004 scripts/submit_alexnet_8192_32.sh
#
#qsub -g gaa50004 scripts/submit_alexnet_16384_64.sh
#
#qsub -g gaa50004 scripts/submit_alexnet_32384_128.sh




## Resnet
#
#qsub -g gaa50004 scripts/submit_resnet_64_8.sh
#qsub -g gaa50004 scripts/submit_resnet_128_16.sh
#qsub -g gaa50004 scripts/submit_resnet_256_32.sh
#sleep 5
#
#
## VGG
#
#qsub -g gaa50004 scripts/submit_vgg_64_8.sh
#qsub -g gaa50004 scripts/submit_vgg_128_16.sh
#qsub -g gaa50004 scripts/submit_vgg_256_32.sh
#sleep 5
#
#

