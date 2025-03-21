#!/bin/bash
python -u src/train.py model=energytemp experiment=lj13_energytemp_debug trainer=gpu model.resampling_interval=1 logger=csv
