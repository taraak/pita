HYDRA_FULL_ERROR=1 python src/train.py +trainer.num_sanity_val_steps=0 \
model=energytemp \
experiment=alp_energytemp \
trainer=ddp model.resampling_interval=1 \
tags=["test","ALDP"] \
model.noise_schedule.sigma_min=0.01 \
trainer.check_val_every_n_epoch=10 \
model.dem.num_training_epochs=0 \
model.debias_inference=False \
model.loss_weights.energy_matching=0.0 \
model.do_energy_matching_loss_every_n_steps=1 \
model.loss_weights.energy_score=0.0 \
model.loss_weights.score=1.0 \
model.loss_weights.target_score=0.00 \
model/net=dit \
model.inference_batch_size=64 \
model.num_samples_to_save=1024 \
model.num_negative_time_steps=1 \
#debug=short \
#model.end_resampling_step=900 \
#model.net.hidden_nf=64 \
#model/net=egnn_dynamics_ad2_cat \
#++model.compile=True
#trainer.gradient_clip_val=100 \
#model.do_langevin=true \
#debug=short \
#+model.only_train_score=True \
#+energy.debug_train_on_test=True \
