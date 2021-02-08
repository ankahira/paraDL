#!/usr/bin/env bash


## Cosmoflow

qsub -g gaa50004 scripts/cosmoflow_all.sh
#
#qsub -g gaa50004 scripts/cosmoflow_4.sh
qsub -g gaa50004 scripts/cosmoflow_8.sh
sleep 100
qsub -g gaa50004 scripts/cosmoflow_16.sh
sleep 100
qsub -g gaa50004 scripts/cosmoflow_32.sh
sleep 100
qsub -g gaa50004 scripts/cosmoflow_64.sh
sleep 100
qsub -g gaa50004 scripts/cosmoflow_128.sh
sleep 100
qsub -g gaa50004 scripts/cosmoflow_256.sh
sleep 100
qsub -g gaa50004 scripts/cosmoflow_512.sh







