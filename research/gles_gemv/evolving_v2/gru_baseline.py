"""
GRU Baseline Model

A simple GRU-based language model for baseline comparison.
Designed to have roughly the same parameter count as the Breathing model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict


@dataclass
class GRUBaselineConfig:
    """Configuration for GRU baseline."""

    vocab_size: int = 50257
    embed_dim: int = 128
    hidden_dim: int = 256  # GRU hidden size
    n_layers: int = 2  # Number of GRU layers
    dropout: float = 0.0


class GRUBaseline(nn.Module):
    """Simple GRU language model baseline."""

    def __init__(self, config: GRUBaselineConfig):
        super().__init__()
        self.config = config

        # Embedding
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim)
        nn.init.normal_(self.embed.weight, std=0.02)

        # GRU layers - takes embed_dim input, outputs embed_dim (for weight tying)
        self.gru = nn.GRU(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.n_layers,
            dropout=config.dropout if config.n_layers > 1 else 0.0,
            batch_first=False,
        )

        # Project GRU output back to embed_dim for weight tying
        self.hidden_to_embed = nn.Linear(
            config.hidden_dim, config.embed_dim, bias=False
        )

        # Output projection (tied to embedding)
        self.out_norm = nn.LayerNorm(config.embed_dim)
        self.out_proj = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.out_proj.weight = self.embed.weight  # Weight tying!

        # State
        self.h = None  # Hidden state
        self.tokens_seen = 0

    def reset_state(self):
        """Reset hidden state."""
        self.h = None

    def forward(self, token_id: int) -> torch.Tensor:
        """
        Forward pass for a single token.

        Args:
            token_id: Token ID

        Returns:
            logits: [vocab_size] logits for next token
        """
        x = self.embed.weight[token_id].unsqueeze(0).unsqueeze(0)  # [1, 1, embed_dim]

        # Detach hidden state to prevent backprop through time across batches
        h = self.h.detach() if self.h is not None else None

        # GRU forward
        output, self.h = self.gru(x, h)

        # Project back to embed_dim and output
        x = self.hidden_to_embed(output.squeeze(0).squeeze(0))  # [embed_dim]
        x = self.out_norm(x)
        logits = self.out_proj(x)

        self.tokens_seen += 1
        return logits

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_stats(self) -> Dict:
        return {
            "tokens_seen": self.tokens_seen,
            "n_params": self.count_parameters(),
            "embed_dim": self.config.embed_dim,
            "hidden_dim": self.config.hidden_dim,
            "n_layers": self.config.n_layers,
        }


def count_breathing_params(
    vocab_size: int = 50257,
    embed_dim: int = 128,
    n_layers: int = 4,
    n_neurons: int = 128,
) -> int:
    """
    Estimate parameter count for Breathing model.

    Parameters:
    - Embedding: vocab_size * embed_dim (shared with output)
    - Per layer:
      - w_up: n_neurons * embed_dim
      - w_gate: n_neurons * embed_dim
      - w_down: n_neurons * embed_dim
      - LayerNorm: 2 * embed_dim
    - Output LayerNorm: 2 * embed_dim
    """
    embed_params = vocab_size * embed_dim
    layer_params = n_neurons * embed_dim * 3 + 2 * embed_dim  # 3 weight matrices + LN
    total_layer_params = layer_params * n_layers
    out_norm_params = 2 * embed_dim

    return embed_params + total_layer_params + out_norm_params


def match_gru_to_breathing(
    vocab_size: int = 50257,
    embed_dim: int = 128,
    n_layers_breathing: int = 4,
    n_neurons: int = 128,
) -> GRUBaselineConfig:
    """
    Create GRU config that matches Breathing model parameter count.
    Now with weight tying, params are much smaller.
    """
    target_params = count_breathing_params(
        vocab_size, embed_dim, n_layers_breathing, n_neurons
    )

    print(f"Target: {target_params:,} params (Breathing model)")

    # Search for best hidden_dim
    best_config = None
    best_diff = float("inf")

    for hidden_dim in range(64, 1024, 16):
        for n_layers in [1, 2, 3]:
            config = GRUBaselineConfig(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
            )
            model = GRUBaseline(config)
            params = model.count_parameters()

            diff = abs(params - target_params)
            if diff < best_diff:
                best_diff = diff
                best_config = config
                best_params = params

    print(
        f"Best match: hidden_dim={best_config.hidden_dim}, n_layers={best_config.n_layers}: {best_params:,} params"
    )
    return best_config


if __name__ == "__main__":
    import pickle
    import time

    print("=" * 60)
    print("GRU Baseline Model Test")
    print("=" * 60)

    # Match parameter count to Breathing model
    print("\nMatching parameters to Breathing model...")
    config = match_gru_to_breathing(
        vocab_size=50257, embed_dim=128, n_layers_breathing=4, n_neurons=128
    )

    model = GRUBaseline(config)
    print(f"\nGRU Config:")
    print(f"  embed_dim: {config.embed_dim}")
    print(f"  hidden_dim: {config.hidden_dim}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  Parameters: {model.count_parameters():,}")

    # Load data
    print("\nLoading WikiText-2...")
    with open("wikitext2_tokens.pkl", "rb") as f:
        tokens = pickle.load(f)[:5000]  # Reduced for faster test
    print(f"Loaded {len(tokens)} tokens")

    # Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("\nTraining...")
    start = time.time()
    total_loss = 0
    correct = 0

    for i in range(len(tokens) - 1):
        logits = model.forward(tokens[i])
        loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([tokens[i + 1]]))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        if logits.argmax().item() == tokens[i + 1]:
            correct += 1

        if (i + 1) % 1000 == 0:
            avg_loss = total_loss / (i + 1)
            acc = correct / (i + 1) * 100
            ppl = min(torch.exp(torch.tensor(avg_loss)).item(), 50000)
            tps = (i + 1) / (time.time() - start)
            print(
                f"Token {i + 1}: Loss={avg_loss:.3f}, PPL={ppl:.0f}, Acc={acc:.2f}%, tok/s={tps:.0f}"
            )

    elapsed = time.time() - start
    final_loss = total_loss / (len(tokens) - 1)
    final_ppl = torch.exp(torch.tensor(final_loss)).item()
    final_acc = correct / (len(tokens) - 1) * 100

    print(f"\nDone in {elapsed:.1f}s ({len(tokens) / elapsed:.0f} tok/s)")
    print(f"Final: Loss={final_loss:.3f}, PPL={final_ppl:.0f}, Acc={final_acc:.2f}%")

    # Quick generation test
    print("\nGeneration test...")
    try:
        from transformers import GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        def generate(prompt, max_tokens=20):
            model.reset_state()
            toks = tokenizer.encode(prompt)
            for t in toks[:-1]:
                model.forward(t)
            for _ in range(max_tokens):
                logits = model.forward(toks[-1])
                probs = F.softmax(logits / 0.8, dim=0)
                toks.append(torch.multinomial(probs, 1).item())
            return tokenizer.decode(toks)

        for p in ["The", "In the", "He said"]:
            print(f"  '{p}' -> {generate(p)}")
    except ImportError:
        print("  (transformers not available for generation test)")

    print("\nGRU Baseline test complete!")
