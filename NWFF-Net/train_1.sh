export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

nohup torchrun --nproc_per_node=2 main.py --config 'config-1.yaml' > train_1.log 2>&1 &

