python src/train.py -m +trainer.num_sanity_val_steps=0 \
model=energytemp \
experiment=lj13_energytemp_debug \
trainer=ddp model.resampling_interval=1 \
tags=["test","LJ13"] \
model.noise_schedule.sigma_min=0.01,0.1,0.05 \
trainer.check_val_every_n_epoch=50 \
