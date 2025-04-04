python -u src/train.py -m +trainer.num_sanity_val_steps=0 model=energytemp experiment=lj13_energytemp_debug trainer=ddp model.resampling_interval=1 tags=["test","LJ13"]
