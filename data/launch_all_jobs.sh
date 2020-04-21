#!/usr/bin/env bash

# Data Parallelism jobs

# Alexnet

#
qsub -g gaa50004 scripts/alexnet/alexnet_2.sh
qsub -g gaa50004 scripts/alexnet/alexnet_4.sh
qsub -g gaa50004 scripts/alexnet/alexnet_8.sh
qsub -g gaa50004 scripts/alexnet/alexnet_16.sh
qsub -g gaa50004 scripts/alexnet/alexnet_32.sh
qsub -g gaa50004 scripts/alexnet/alexnet_64.sh

qsub -g gaa50004 scripts/alexnet/alexnet_128.sh
qsub -g gaa50004 scripts/alexnet/alexnet_256.sh
qsub -g gaa50004 scripts/alexnet/alexnet_512.sh
qsub -g gaa50004 scripts/alexnet/alexnet_1024.sh


### VGG

qsub -g gaa50004 scripts/vgg/vgg_2.sh
qsub -g gaa50004 scripts/vgg/vgg_4.sh
qsub -g gaa50004 scripts/vgg/vgg_8.sh
qsub -g gaa50004 scripts/vgg/vgg_16.sh
qsub -g gaa50004 scripts/vgg/vgg_32.sh
qsub -g gaa50004 scripts/vgg/vgg_64.sh

qsub -g gaa50004 scripts/vgg/vgg_128.sh
qsub -g gaa50004 scripts/vgg/vgg_256.sh
qsub -g gaa50004 scripts/vgg/vgg_512.sh
qsub -g gaa50004 scripts/vgg/vgg_1024.sh

#
#
#
#
### Resnet
###

qsub -g gaa50004 scripts/resnet/resnet_2.sh
qsub -g gaa50004 scripts/resnet/resnet_4.sh
qsub -g gaa50004 scripts/resnet/resnet_8.sh
qsub -g gaa50004 scripts/resnet/resnet_16.sh
qsub -g gaa50004 scripts/resnet/resnet_32.sh
qsub -g gaa50004 scripts/resnet/resnet_64.sh

qsub -g gaa50004 scripts/resnet/resnet_128.sh
qsub -g gaa50004 scripts/resnet/resnet_256.sh
qsub -g gaa50004 scripts/resnet/resnet_512.sh
qsub -g gaa50004 scripts/resnet/resnet_1024.sh




