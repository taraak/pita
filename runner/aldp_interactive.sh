HYDRA_FULL_ERROR=1 python src/train.py +trainer.num_sanity_val_steps=0 \
model=energytemp \
experiment=alp_energytemp \
trainer=ddp model.resampling_interval=1 \
tags=["test","ALDP","no_lambda_target_score"] \
model.noise_schedule.sigma_min=0.01 \
trainer.check_val_every_n_epoch=50 \
model.dem.num_training_epochs=0 \
model.debias_inference=True \
model.resample_at_end=False \
model.loss_weights.energy_matching=1.0 \
model.loss_weights.energy_score=1.0 \
model.loss_weights.score=1.0 \
model.loss_weights.target_score=0.01 \
model.loss_time_threshold.score=0 \
model.inference_batch_size=512 \
model.training_batch_size=1024 \
model.num_negative_time_steps=0 \
model.post_mcmc_steps=20 \
model.end_resampling_step=800 \
model/net=dit \
#debug=short \
#+model.only_train_score=True \
#+energy.debug_train_on_test=True \
#debug=short \
