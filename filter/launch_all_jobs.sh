#!/usr/bin/env bash

# Filter Parallelism jobs

# Alexnet


qsub -g gaa50004 scripts/submit_alexnet_128_4.sh
sleep 5
qsub -g gaa50004 scripts/submit_alexnet_256_4.sh
sleep 5
qsub -g gaa50004 scripts/submit_alexnet_512_4.sh
sleep 5




#qsub -g gaa50004 scripts/submit_alexnet.sh
#qsub -g gaa50004 scripts/submit_vgg.sh
#qsub -g gaa50004 scripts/submit_resnet.sh
#sleep 5
#
#
