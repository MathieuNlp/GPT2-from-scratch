import torch
import tiktoken


class DataLoaderLite():
    def __init__(self, B, T):
        self.B = B
        self.T = T
    
        # at init load tokens from disk and store them in memory
        with open("./data/input.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1] # need the +1 for the target tokens
        x = (buf[:-1].view(B, T)) # inputs
        y = (buf[1:]).view(B, T) # targets
        # adevance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T  + 1) > len(self.tokens):
            self.current_position = 0
        
        return x, y
        
if __name__ == "__main__":
    from model.gpt2 import GPT
    from config.config import GPTConfig
    import math
    import time
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # total_batch_size = 524288 # 2**19, ~0.5M, in numbers of tokens => 0.5M is the batch size from GPT-3 paper, where batch size is B*T
    total_batch_size = 16_384 # Reduce the total batch size for low memory => Otherwise too long to do all the accumulations
    B = 4 # micro batch size
    T = 256 # sequence length

    assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
    grad_accum_steps = total_batch_size // (B * T)
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    train_loader = DataLoaderLite(B=B, T=T)

    torch.set_float32_matmul_precision('high') # Allows to change the precsion from FP32 to TF32

    model = GPT(GPTConfig(vocab_size=50304)) # Use a "nice" number for the vocab size => divisible by 2^x
    model.to(device)
    model = torch.compile(model)
    
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

    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

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
            loss.backward() # Inside loss.backward() there is always a += so we can use gradient acucmulation
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Used in GPT-3 that we brought back in GPT-2
        # determine and set the learning for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        torch.cuda.synchronize() # wait for GPU to finish work
        t1 = time.time()
        dt = (t1 - t0) # time difference in seconds
        tokens_processed = (train_loader.B * train_loader.T) * grad_accum_steps
        tokens_per_sec = tokens_processed / dt
        tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
        print(f"step {step:4d} | loss {loss_accum.item():.6f} | lr {lr:.4e} | norm gradient: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")