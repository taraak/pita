
HYDRA_FULL_ERROR=1 python -u src/train.py \
+trainer.num_sanity_val_steps=0 \
model=energytemp \
experiment=alp_energytemp_good \
trainer=ddp model.resampling_interval=1 \
tags=["ALDP","dit","wise_silence","temp_cond"] \
model/net=dit \
model.noise_schedule.sigma_min=0.005 \
model.dem.num_training_epochs=0 \
trainer.check_val_every_n_epoch=100 \
model.dem.num_training_epochs=0 \
model.debias_inference=True \
model.resample_at_end=False \
model.loss_weights.energy_matching=1.0 \
model.loss_weights.energy_score=1.0 \
model.loss_weights.score=1.0 \
model.loss_weights.target_score=0.01 \
model.loss_time_threshold.score=0 \
model.inference_batch_size=512 \
model.num_negative_time_steps=0 \
model.do_langevin=False \
model.post_mcmc_steps=5 \
model.loss_time_threshold.score=0 \
model.end_resampling_step=800 \
model.num_temp_annealed_samples_to_generate=10000 \
model.num_epochs_per_temp=[300,300,300,300,300] \
