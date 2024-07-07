import os
import math
import time
import tiktoken
import torch
from torch.nn import functional as F
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from model.gpt2 import GPT
from config.config import GPTConfig
from dataloader.train_dataloader import DataLoaderLite


# simple launch:
# python3 train_gpt2.py
# DDP launch for e.g 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py


# set up DDP (destributed data parallel)
# torchrun command sets the env variables RANK, LOCAL_RANK and WORLD_SIZE !
# We can imagine the each process will run it's own script with different env variables

ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now I think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_local_rank == 0 # this process (the process cuda:0) will do logging, checkpointing etc
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

torch.manual_seed(1337) # Because the seed is fixed, we create the same identical model in DDP
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337) 

#total_batch_size = 524288 # 2**19, ~0.5M, in numbers of tokens => 0.5M is the batch size from GPT-3 paper, where batch size is B*T
total_batch_size = 32768 # 2**15, Reduce the total batch size for low memory => Otherwise too long to do all the accumulations
B = 2 # micro batch size
T = 1024 # sequence length

assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B,
                              T=T,
                              process_rank=ddp_rank,
                              num_processes=ddp_world_size,
                              master_process=master_process,
                              split='train'
                              )

val_loader = DataLoaderLite(B=B,
                              T=T,
                              process_rank=ddp_rank,
                              num_processes=ddp_world_size,
                              master_process=master_process,
                              split='val'
                              )
torch.set_float32_matmul_precision('high') # Allows to change the precsion from FP32 to TF32

# Create model
model = GPT(GPTConfig(vocab_size=50304)) # Use a "nice" number for the vocab size => divisible by 2^x
model.to(device)
model = torch.compile(model)
if ddp:
    # DDP class can distribute the gradient during the backward pass => More efficient
    # It can also apply all reduce operation to gather the gradients from all the ranks, average and update the parameters (<=> synchronizing the gradient)
    model = DDP(model, device_ids=[ddp_local_rank]) 
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model to use it's functions (e.g configure_optimizers)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10681  # In GPT-3 paper, warmup is for the first 350M tokens => 350M/2**19 = 715
max_steps = 30518 # I am using 10% of 10B dataset => 1B/2**19 = 1907. If all 10B dataset, use 19073

def get_lr(it):
    """Learning rate setup for the training (from GPT-3 paper but same as GPT-2)"""
    # 1) linear warmup for "warmup_steps" steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps # (it+1), the +1 is such that at it=0, we don't have lr=0
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

for step in range(max_steps):
    t0 = time.time()

    # once in a while evaluate our validation loss
    if step % 100 == 0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16): # only put the forward pass in the autocast (optimizer and backprop is outside)
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach() 
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")

    # once in a while generate from the model (except step 0, which is noise)
    if step > 0 and step % 100 == 0:
        model.eval()
        num_return_sequences = 4
        max_length = 32
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long) # torch.long is equivalent of self.to(torch.int64)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                # take the logits at the last position
                logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits of the last position
                logits = logits[:, -1, :]
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demande the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # training loop
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16): # only put the forward pass in the autocast (optimizer and backprop is outside)
            logits, loss = model(x, y)
        # we have to scale the loss to account for the gradient accumulation,
        # because the gradients just add on each succesive backward(),
        # addition of graddients correspond to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            # We are using gradient accumulation and we don't want to synchronize the gradients each time in DDP,
            # we only synchronize when we are at the last step of accumulation
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward() # Inside loss.backward() there is always a += so we can use gradient acucmulation
    if ddp:
        # We are only printing on the master_process (cuda:0),
        # so we need to do all reduce to gather the gradients from all processes,
        # to print it correctly.
        # After all_reduce, all the processes will have the same loss_accum averaged
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Used in GPT-3 that we brought back in GPT-2
    # determine and set the learning for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize() # wait for GPU to finish work
    t1 = time.time()
    dt = (t1 - t0) # time difference in seconds
    tokens_processed = (train_loader.B * train_loader.T) * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:4d} | loss {loss_accum.item():.6f} | lr {lr:.4e} | norm gradient: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

if ddp:
    destroy_process_group()