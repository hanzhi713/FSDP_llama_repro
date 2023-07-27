# Llama with Pytorch FSDP

## Install dependencies

```bash
pip install --pre torch==2.1.0.dev20230727 torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
pip install transformers tqdm
```

## Run

Local, single node

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=8 train_gpu_open.py
```

Multi node

```bash
torchrun \
  --nproc_per_node=8 \
  --nnodes=16 \
  --rdzv_id=100 \
  --rdzv_backend=c10d \
  --rdzv_conf=timeout=9000,read_timeout=9000 \
  --rdzv_endpoint=IP \
  --role=torch-worker \
  train_gpu_open.py
```