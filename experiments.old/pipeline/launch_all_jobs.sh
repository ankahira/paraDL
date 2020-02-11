#!/usr/bin/env bash

# Data Pipeline jobs

qsub -g gaa50004 scripts/alexnet/submit_alexnet_32_4.sh
qsub -g gaa50004 scripts/alexnet/submit_alexnet_64_4.sh
qsub -g gaa50004 scripts/alexnet/submit_alexnet_128_4.sh


