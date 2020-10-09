#!/usr/bin/env bash

# Channel Parallelism jobs

# Alexnet

### Resnet

qsub -g gaa50004 scripts/resnet/resnet_4.sh
qsub -g gaa50004 scripts/resnet/resnet_8.sh
qsub -g gaa50004 scripts/resnet/resnet_16.sh
qsub -g gaa50004 scripts/resnet/resnet_32.sh
qsub -g gaa50004 scripts/resnet/resnet_64.sh
