#!/bin/bash
python -u src/train.py model=energytemp +trainer.num_sanity_val_steps=0 experiment=lj13_energytemp trainer=gpu model.resampling_interval=1 logger=csv
