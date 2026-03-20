"""Quick train - 50 steps just to get a checkpoint for testing."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../bdh'))

import torch
import torch.nn.functional as F
from bdh import BDH, BDHConfig
import numpy as np

device = torch.device('cpu')
print(f'Using device: {device}')

config = BDHConfig(
    n_layer=4, n_embd=128, dropout=0.1,
    n_head=2, mlp_internal_dim_multiplier=8, vocab_size=256,
)

PROMPTS = ['2+2=4', 'How are you ?', 'The capital of France is Paris']

data = []
for p in PROMPTS:
    data.append(torch.tensor(bytearray(p, 'utf-8'), dtype=torch.long))

model = BDH(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

NUM_STEPS = 50
model.train()
for step in range(NUM_STEPS):
    bx, by = [], []
    for _ in range(8):
        tokens = data[np.random.randint(0, len(data))]
        if len(tokens) > 1:
            bx.append(tokens[:-1])
            by.append(tokens[1:])
    ml = max(len(x) for x in bx)
    bx = torch.stack([F.pad(x, (0, ml - len(x))) for x in bx]).to(device)
    by = torch.stack([F.pad(y, (0, ml - len(y))) for y in by]).to(device)
    logits, loss = model(bx, by)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 10 == 0:
        print(f'Step {step:3d}/{NUM_STEPS}  Loss: {loss.item():.4f}')

print('\nDone! Testing generation...')
model.eval()
with torch.no_grad():
    for prompt in ['2+2=', 'How are', 'The capital']:
        tok = torch.tensor(bytearray(prompt, 'utf-8'), dtype=torch.long, device=device).unsqueeze(0)
        gen = model.generate(tok, max_new_tokens=8, temperature=0.5, top_k=5)
        txt = bytes(gen[0].tolist()).decode(errors='replace')
        print(f"  '{prompt}' -> '{txt}'")

save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bdh_trained.pt')
torch.save({'model_state_dict': model.state_dict(), 'config': config}, save_path)
print(f'\nModel saved to {save_path}')
