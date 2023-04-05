#!/bin/bash

cd src
	
# Run experiment
# Run experiment
python3 main.py \
    --cpus-per-trial 4 \
    --dataset cifar10 \
    --project-name xps_uncert
# python3 main.py  --cpus-per-trial 4 --dataset mnist --project-name aal_base_AfterDark
