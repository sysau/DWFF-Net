export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

nohup torchrun --nproc_per_node=2 main.py --config 'config-2.yaml' > train_2.log 2>&1 &

