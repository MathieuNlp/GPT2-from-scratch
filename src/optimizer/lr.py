import math

def get_lr(it, min_lr, max_lr, warmup_steps, max_steps):
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