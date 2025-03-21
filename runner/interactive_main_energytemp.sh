srun python -u src/train.py -m model=energytemp experiment=lj13_energytemp_debug trainer=ddp model.resampling_interval=1 tags=["test","LJ13"]

