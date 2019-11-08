#!/usr/bin/env bash

# Channel Parallelism jobs



qsub -g gaa50004 scripts/alexnet/alexnet_4.sh

#qsub -g gaa50004 scripts/submit_vgg.sh
#qsub -g gaa50004 scripts/submit_resnet.sh
#sleep 5
#
## Data Parallelism jobs
#
#cd /home/acb10954wf/parallelism-in-deep-learning/data
#qsub -g gaa50004 scripts/submit_alexnet.sh
#qsub -g gaa50004 scripts/submit_vgg.sh
#qsub -g gaa50004 .scripts/submit_resnet.sh
#sleep 5
#
#
## Filter Parallelism jobs
#
#cd /home/acb10954wf/parallelism-in-deep-learning/filter
#qsub -g gaa50004 scripts/submit_alexnet.sh
#qsub -g gaa50004 scripts/submit_vgg.sh
#qsub -g gaa50004 scripts/submit_resnet.sh
#sleep 5
#
#
