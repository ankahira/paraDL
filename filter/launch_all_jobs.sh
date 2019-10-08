#!/usr/bin/env bash

# Filter Parallelism jobs

qsub -g gaa50004 scripts/submit_alexnet.sh
qsub -g gaa50004 scripts/submit_vgg.sh
qsub -g gaa50004 scripts/submit_resnet.sh
sleep 5

