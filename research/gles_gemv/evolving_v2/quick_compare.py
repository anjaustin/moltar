#!/usr/bin/env python3
"""Quick comparison - writes results to file."""

import torch
import torch.nn.functional as F
import pickle
import time
import json

from breathing_model import BreathingModel, BreathingModelConfig
from gru_baseline import GRUBaseline, GRUBaselineConfig

N = 5000
results = {"n_tokens": N}

with open("wikitext2_clean_tokens.pkl", "rb") as f:
    tokens = pickle.load(f)[: N + 1]

# Breathing
b_model = BreathingModel(
    BreathingModelConfig(
        n_neurons_per_layer=128, n_layers=4, n_breathe=5, evolve_every=500
    )
)
results["breathing_params"] = sum(p.numel() for p in b_model.parameters())

optimizer = torch.optim.AdamW(b_model.parameters(), lr=1e-3)
start = time.time()
total_loss = 0
correct = 0
for i in range(N):
    logits = b_model.forward(tokens[i])
    loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([tokens[i + 1]]))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(b_model.parameters(), 1.0)
    optimizer.step()
    total_loss += loss.item()
    if logits.argmax().item() == tokens[i + 1]:
        correct += 1
    b_model.maybe_evolve()

results["breathing_loss"] = total_loss / N
results["breathing_ppl"] = torch.exp(torch.tensor(results["breathing_loss"])).item()
results["breathing_acc"] = correct / N * 100
results["breathing_time"] = time.time() - start

# GRU
g_model = GRUBaseline(GRUBaselineConfig(hidden_dim=96, n_layers=3))
results["gru_params"] = sum(p.numel() for p in g_model.parameters())

optimizer = torch.optim.AdamW(g_model.parameters(), lr=1e-3)
start = time.time()
total_loss = 0
correct = 0
for i in range(N):
    logits = g_model.forward(tokens[i])
    loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([tokens[i + 1]]))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(g_model.parameters(), 1.0)
    optimizer.step()
    total_loss += loss.item()
    if logits.argmax().item() == tokens[i + 1]:
        correct += 1

results["gru_loss"] = total_loss / N
results["gru_ppl"] = torch.exp(torch.tensor(results["gru_loss"])).item()
results["gru_acc"] = correct / N * 100
results["gru_time"] = time.time() - start

# Save
with open("experiments/quick_comparison_5k.json", "w") as f:
    json.dump(results, f, indent=2)

print("Done! Results saved to experiments/quick_comparison_5k.json")
