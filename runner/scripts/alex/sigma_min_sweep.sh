python src/train.py -m +trainer.num_sanity_val_steps=0 \
launcher=mila_a100_yb \
model=energytemp \
experiment=lj55_energytemp_debug \
trainer=ddp model.resampling_interval=1 \
tags=["test","LJ13","sigma_min_sweep"] \
model.noise_schedule.sigma_min=0.002,0.005,0.05,0.01,0.1 \
trainer.check_val_every_n_epoch=50 \
#experiment=lj13_energytemp_debug \
