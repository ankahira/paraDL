#!/usr/bin/env bash

## Filter Hybrid  Parallelism jobs

qsub -g gaa50004 scripts/resnet/resnet_8.sh
qsub -g gaa50004 scripts/resnet/resnet_16.sh
qsub -g gaa50004 scripts/resnet/resnet_32.sh
qsub -g gaa50004 scripts/resnet/resnet_64.sh
qsub -g gaa50004 scripts/resnet/resnet_128.sh
qsub -g gaa50004 scripts/resnet/resnet_256.sh
qsub -g gaa50004 scripts/resnet/resnet_512.sh
qsub -g gaa50004 scripts/resnet/resnet_1024.sh
