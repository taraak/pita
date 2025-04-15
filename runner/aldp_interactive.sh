python src/train.py -m +trainer.num_sanity_val_steps=0 \
model=energytemp \
experiment=alp_energytemp \
trainer=ddp model.resampling_interval=1 \
tags=["test","ALDP"] \
model.noise_schedule.sigma_min=0.05 \
data.n_train_batches_per_epoch=250 \
debug=short \

