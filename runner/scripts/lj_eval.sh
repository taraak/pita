

python src/eval.py -m \
'ckpt_path="/network/scratch/t/tara.akhoundsadegh/dem/logs/train/runs/2024-01-25_21-42-58/active-inference/mp6d9o5z/checkpoints/epoch=999-step=100000.ckpt"' \
model.num_integration_steps=1,10,20,50,100,200,500,1000 \
model.nll_integration_method=euler \
experiment=lj13 \
tags=["eval_sweep_v4","lj13"]

python src/eval.py -m \
'ckpt_path="/network/scratch/t/tara.akhoundsadegh/dem/logs/train/runs/2024-01-25_21-42-58/active-inference/mp6d9o5z/checkpoints/epoch=999-step=100000.ckpt"' \
model.tol=1e-2,1e-3,1e-4,1e-5,1e-6,1e-7 \
model.nll_integration_method=dopri5 \
experiment=lj13 \
tags=["eval_sweep_v4","lj13"]
