#!/usr/bin/env bash

## Filter Hybrid  Parallelism jobs

### VGG

#qsub -g gaa50004 scripts/vgg/vgg_8.sh
#qsub -g gaa50004 scripts/vgg/vgg_16.sh
#qsub -g gaa50004 scripts/vgg/vgg_32.sh
#qsub -g gaa50004 scripts/vgg/vgg_64.sh
#qsub -g gaa50004 scripts/vgg/vgg_128.sh
#qsub -g gaa50004 scripts/vgg/vgg_256.sh
#qsub -g gaa50004 scripts/vgg/vgg_512.sh
#qsub -g gaa50004 scripts/vgg/vgg_1024.sh


# ResNet50


#qsub -g gaa50004 scripts/resnet50/resnet_8.sh
#qsub -g gaa50004 scripts/resnet50/resnet_16.sh
#qsub -g gaa50004 scripts/resnet50/resnet_32.sh
#qsub -g gaa50004 scripts/resnet50/resnet_64.sh
#qsub -g gaa50004 scripts/resnet50/resnet_128.sh
#qsub -g gaa50004 scripts/resnet50/resnet_256.sh
#qsub -g gaa50004 scripts/resnet50/resnet_512.sh
#qsub -g gaa50004 scripts/resnet50/resnet_1024.sh



## ResNet152
#
#qsub -g gaa50004 scripts/resnet152/resnet_8.sh
qsub -g gaa50004 scripts/resnet152/resnet_16.sh
qsub -g gaa50004 scripts/resnet152/resnet_32.sh
qsub -g gaa50004 scripts/resnet152/resnet_64.sh
qsub -g gaa50004 scripts/resnet152/resnet_128.sh
qsub -g gaa50004 scripts/resnet152/resnet_256.sh
qsub -g gaa50004 scripts/resnet152/resnet_512.sh
qsub -g gaa50004 scripts/resnet152/resnet_1024.sh