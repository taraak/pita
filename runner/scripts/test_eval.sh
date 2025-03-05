

#python src/eval.py -m \
#'ckpt_path="/network/scratch/a/alexander.tong/dem/logs/train/runs/2024-01-27_22-14-20/active-inference/j669zuzw/checkpoints/epoch=369-step=37000.ckpt"' \
#model.tol=1e-2,1e-3,1e-4,1e-5,1e-6,1e-7 \
#model.nll_integration_method=dopri5 \
#tags=["eval_sweep_v3"]

python src/eval.py -m \
'ckpt_path="/network/scratch/a/alexander.tong/dem/logs/train/runs/2024-01-27_22-14-20/active-inference/j669zuzw/checkpoints/epoch=369-step=37000.ckpt"' \
model.num_integration_steps=1,10,20,50,100,200,500,1000 \
model.nll_integration_method=euler \
tags=["eval_sweep_v3"]
