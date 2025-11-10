export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

nohup torchrun --nproc_per_node=2 main.py --config 'config-3.yaml' > train_3.log 2>&1 &

