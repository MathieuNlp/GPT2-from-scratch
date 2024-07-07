import torch
import tiktoken


class DataLoaderLite():
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
    
        # at init load tokens from disk and store them in memory
        with open("./data/input.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = self.B * self.T * self.process_rank # For each process in the DDP, we start at a different position. We cover a large chunk

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1] # need the +1 for the target tokens
        x = (buf[:-1].view(B, T)) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes # We make a big jump because the other processes already consumed a chunk of data
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        
        return x, y
        
if __name__ == "__main__":
    import os
    from model.gpt2 import GPT
    from config.config import GPTConfig
    import math
    import time
    import torch.distributed as dist
    from torch.distributed import init_process_group, destroy_process_group
    from torch.nn.parallel import DistributedDataParallel as DDP

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

    # total_batch_size = 524288 # 2**19, ~0.5M, in numbers of tokens => 0.5M is the batch size from GPT-3 paper, where batch size is B*T
    total_batch_size = 16384 # Reduce the total batch size for low memory => Otherwise too long to do all the accumulations
    B = 4 # micro batch size
    T = 256 # sequence length

    assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)

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
    warmup_steps = 10
    max_steps = 50

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