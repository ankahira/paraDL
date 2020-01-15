#!/usr/bin/env bash

# Spatial  Parallelism jobs


# Debug jobs
#
#qsub -g gaa50004 scripts/alexnet/alexnet_debug.sh
#
#qsub -g gaa50004 scripts/vgg/vgg_debug.sh
#
#qsub -g gaa50004 scripts/resnet/resnet_debug.sh
#


# 4 GPU runs
# These runs are not any different from the debug runs. Just the directory for results changes
 # Ennsure that during this runs, validation is not activated. so set a high number for validation interval


qsub -g gaa50004 scripts/alexnet/alexnet_4.sh

qsub -g gaa50004 scripts/vgg/vgg_4.sh

qsub -g gaa50004 scripts/resnet/resnet_4.sh


#
## Full jobs
# # These jobs must include validation

## These run to 70 epochs
## Make sure the time assigned is enough
#
#qsub -g gaa50004 scripts/alexnet/alexnet_full.sh
#
#qsub -g gaa50004 scripts/vgg/vgg_full.sh
#
#qsub -g gaa50004 scripts/resnet/resnet_full.sh







