python src/train.py -m experiment=lj13_2 model.use_buffer=False,True model.noise_schedule.sigma_max=0.05,0.1,0.2,0.3,0.4,0.5,1,2,5,10 launcher=mila_cluster tags=["LJ13-2","exp_buffer_1"] logger.wandb.group="lj13-3"
#python src/train.py -m experiment=gmm,lj13 model.use_buffer=False,True model.noise_schedule.sigma_max=0.05,0.1,0.2,0.3,0.4,0.5,1,2,5,10 launcher=mila_cluster 
