HYDRA_FULL_ERROR=1 python src/train.py +trainer.num_sanity_val_steps=0 \
model=energytemp \
experiment=al3_energytemp_good \
trainer=ddp model.resampling_interval=1 \
tags=["test","AL3"] \
trainer.check_val_every_n_epoch=100 \
model.debias_inference=True \
model.resample_at_end=True \
model.loss_weights.energy_matching=1.0 \
model.loss_weights.energy_score=1.0 \
model.loss_weights.score=1.0 \
model.loss_weights.target_score=0.01 \
model.loss_time_threshold.score=0 \
model.inference_batch_size=512 \
model.training_batch_size=1024 \
model.num_negative_time_steps=0 \
model.post_mcmc_steps=5 \
model.end_resampling_step=800 \
model/net=dit \
#model.net.hidden_size=768 \
#debug=short \
#+model.only_train_score=True \
#+energy.debug_train_on_test=True \
#debug=short \
