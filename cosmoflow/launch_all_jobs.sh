#!/usr/bin/env bash

# Data Parallelism jobs

# Cosmoflow

qsub -g gaa50004 scripts/cosmoflow_4.sh
qsub -g gaa50004 scripts/cosmoflow_8.sh
sleep 100
qsub -g gaa50004 scripts/cosmoflow_16.sh
sleep 100
qsub -g gaa50004 scripts/cosmoflow_32.sh
sleep 500
qsub -g gaa50004 scripts/cosmoflow_64.sh
sleep 500
qsub -g gaa50004 scripts/cosmoflow_128.sh
sleep 600
qsub -g gaa50004 scripts/cosmoflow_256.sh
sleep 600
qsub -g gaa50004 scripts/cosmoflow_512.sh






