import torch
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
import sys
from torch.optim import AdamW
from tqdm.auto import tqdm

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import os
from functools import partial
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_iters", type=int, default=10000)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--params", type=str, default="7b")
parser.add_argument("--prof_dir", type=str, default="llama_log")


args = parser.parse_args()

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
dist.init_process_group(backend="nccl", 
                        rank=rank,
                        world_size=world_size)
torch.cuda.set_device(local_rank)
dev = torch.cuda.current_device()

def master_print(*args):
    if local_rank == 0:
        print(*args, flush=True)

def get_model(config_path):
    strategy = ShardingStrategy.FULL_SHARD
    with open(config_path) as f:
        config = LlamaConfig(**json.load(f))

    if rank == 0:
        master_print("rank 0 init model...")
        model = LlamaForCausalLM(config)
        master_print("rank 0 init model done")
    else:
        with torch.device("meta"):
            model = LlamaForCausalLM(config)

    model = FSDP(model,
        sharding_strategy=strategy,
        sync_module_states=True,
        mixed_precision=MixedPrecision(param_dtype=torch.float16),
        auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={LlamaDecoderLayer}),
        limit_all_gathers=True,
        device_id=dev,
        param_init_fn=lambda x: x.to_empty(device=dev, recurse=False)
    )
    return model


def main():
    model = get_model(f"config_{args.params}.json")
    optimizer = AdamW(model.parameters(), lr=3.0e-4)
    
    batch_size = args.batch_size
    p_output = sys.stdout if rank == 0 else open(os.devnull, "w")
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(args.prof_dir),
        record_shapes=True,
        with_stack=True) as prof:
        for step in tqdm(range(args.num_iters), file=p_output, smoothing=0.95):
            input_ids = torch.ones((batch_size, 2048), device=dev, dtype=torch.int64)
            optimizer.zero_grad()
            loss = model(input_ids=input_ids, labels=input_ids).loss
            loss.backward()
            optimizer.step()
            prof.step()

            if step % 5 == 0:
                with torch.no_grad():
                    avg_loss = loss.detach().data.clone()
                    dist.all_reduce(avg_loss, dist.ReduceOp.AVG)
                    master_print(step, "loss ", avg_loss.item())


if __name__ == "__main__":
    main()