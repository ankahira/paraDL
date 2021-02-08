#!/usr/bin/env bash

# Channel Parallelism jobs


### Resnet50

#qsub -g gaa50004 scripts/resnet50/resnet_2.sh
#qsub -g gaa50004 scripts/resnet50/resnet_4.sh
qsub -g gaa50004 scripts/resnet50/resnet_8.sh
qsub -g gaa50004 scripts/resnet50/resnet_16.sh
qsub -g gaa50004 scripts/resnet50/resnet_32.sh
qsub -g gaa50004 scripts/resnet50/resnet_64.sh


### Resnet152

#qsub -g gaa50004 scripts/resnet152/resnet_2.sh
#qsub -g gaa50004 scripts/resnet152/resnet_3.sh
#qsub -g gaa50004 scripts/resnet152/resnet_4.sh
qsub -g gaa50004 scripts/resnet152/resnet_8.sh
qsub -g gaa50004 scripts/resnet152/resnet_16.sh
qsub -g gaa50004 scripts/resnet152/resnet_32.sh
qsub -g gaa50004 scripts/resnet152/resnet_64.sh

### VGG

#qsub -g gaa50004 scripts/vgg/vgg_2.sh
#qsub -g gaa50004 scripts/vgg/vgg_4.sh
qsub -g gaa50004 scripts/vgg/vgg_8.sh
qsub -g gaa50004 scripts/vgg/vgg_16.sh
qsub -g gaa50004 scripts/vgg/vgg_32.sh
qsub -g gaa50004 scripts/vgg/vgg_64.sh
