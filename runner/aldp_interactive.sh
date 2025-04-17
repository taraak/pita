HYDRA_FULL_ERROR=1 python src/train.py +trainer.num_sanity_val_steps=0 \
model=energytemp \
experiment=alp_energytemp \
trainer=ddp model.resampling_interval=-1 \
tags=["test","ALDP"] \
model.noise_schedule.sigma_min=0.0005 \
trainer.check_val_every_n_epoch=50 \
model.dem.num_training_epochs=0 \
model.debias_inference=False \
model.loss_weights.energy_matching=0.0 \
model.loss_weights.energy_score=0.0 \
#model/net=egnn_dynamics_ad2_cat \
#+model.only_train_score=True \
#+energy.debug_train_on_test=True \
#debug=short \

