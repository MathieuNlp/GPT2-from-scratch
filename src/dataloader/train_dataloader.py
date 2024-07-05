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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    B, T = 4, 32
    train_loader = DataLoaderLite(B, T)
    model = GPT(GPTConfig())
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for i in range(50):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        import code; code.interact(local=locals())
        loss.backward()
        optimizer.step()
        print(f"step {i}, loss {loss.item()}")