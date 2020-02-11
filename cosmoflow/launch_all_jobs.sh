#!/usr/bin/env bash

# Data Parallelism jobs

# Cosmoflow

qsub -g gaa50004 scripts/cosmoflow_4.sh
qsub -g gaa50004 scripts/cosmoflow_8.sh
qsub -g gaa50004 scripts/cosmoflow_16.sh
#qsub -g gaa50004 scripts/cosmoflow_32.sh
#qsub -g gaa50004 scripts/cosmoflow_64.sh
#qsub -g gaa50004 scripts/cosmoflow_128.sh
#qsub -g gaa50004 scripts/cosmoflow_256.sh
#qsub -g gaa50004 scripts/cosmoflow_512.sh







