HYDRA_FULL_ERROR=1 python src/train.py +trainer.num_sanity_val_steps=0 \
model=energytemp \
experiment=alp_energytemp \
trainer=ddp model.resampling_interval=1 \
tags=["test","ALDP"] \
model.noise_schedule.sigma_min=0.01 \
trainer.check_val_every_n_epoch=50 \
model.dem.num_training_epochs=0 \
model.debias_inference=True \
model.loss_weights.energy_matching=1.0 \
model.do_energy_matching_loss_every_n_steps=1 \
model.loss_weights.energy_score=1.0 \
model.loss_weights.score=1.0 \
model.loss_weights.target_score=0.0 \
model/net=egnn_dynamics_ad2_cat \
model.inference_batch_size=128 \
model.num_negative_time_steps=1 \
model.end_resampling_step=900 \
model.net.hidden_nf=64 \
#debug=short \
#trainer.gradient_clip_val=100 \
#model.do_langevin=true \
#debug=short \
#+model.only_train_score=True \
#+energy.debug_train_on_test=True \
