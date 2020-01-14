#!/usr/bin/env bash

# Channel Parallelism jobs

# Single epoch jobs

#
#qsub -g gaa50004 scripts/alexnet/alexnet_3.sh
#
#qsub -g gaa50004 scripts/vgg/vgg_3.sh
#
#qsub -g gaa50004 scripts/resnet/resnet_3.sh


# full jobs

qsub -g gaa50004 scripts/alexnet/alexnet_full.sh

qsub -g gaa50004 scripts/vgg/vgg_full.sh

qsub -g gaa50004 scripts/resnet/resnet_full.sh




