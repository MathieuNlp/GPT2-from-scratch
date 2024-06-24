import torch
import tiktoken

from model.gpt2 import GPT
from config.config import GPTConfig


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {device}")

# get a batch
enc = tiktoken.get_encoding('gpt2')
with open("./data/input.txt", "r") as f:
    text = f.read()
text = text[:1000]
tokens = enc.encode(text)
B, T = 4, 32
buf = torch.tensor(tokens[:B*T + 1 ])
x = buf[:-1].view(B, T).to(device)
y = buf[1:].view(B, T)

print(y)
# get logits
model = GPT(GPTConfig())
model.to(device)
logits, loss = model(x, y)

print(loss)
import sys; sys.exit(0)

# prefix tokens