#!/usr/bin/env python3
"""
Create a small test LFM model for benchmarking SpaceGhost optimizations
"""

import torch
import torch.nn as nn
import json
from pathlib import Path

class TestLFMModel(nn.Module):
    """Small LFM-like model for testing SpaceGhost optimizations"""

    def __init__(self, vocab_size=32000, hidden_size=768, num_layers=6, num_heads=12):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layer
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # Transformer layers (simplified LFM architecture)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        # Output projection
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids):
        # Embed tokens
        x = self.embed_tokens(input_ids)

        # Create causal mask for autoregressive generation
        seq_len = input_ids.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(x.device)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, x, tgt_mask=causal_mask)

        # Project to vocabulary
        logits = self.lm_head(x)
        return logits

def create_test_lfm_model(model_name="TestLFM-25M", save_path="models/test_lfm"):
    """Create and save a small test LFM model"""

    print(f"🔧 Creating {model_name} model...")
    print(f"   Save path: {save_path}")

    # Create model directory
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # Create model
    model = TestLFMModel(
        vocab_size=32000,
        hidden_size=512,  # Smaller for testing
        num_layers=4,     # Fewer layers
        num_heads=8
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(",.1f")

    # Create sample input
    sample_input = torch.randint(0, 32000, (1, 32))  # Batch size 1, sequence length 32

    # Test forward pass
    print("🧠 Testing model forward pass...")
    model.eval()
    with torch.no_grad():
        output = model(sample_input)
        print(f"   Input shape: {sample_input.shape}")
        print(f"   Output shape: {output.shape}")

    # Save model
    print("💾 Saving model...")
    torch.save(model.state_dict(), f"{save_path}/pytorch_model.bin")

    # Create config files
    config = {
        "model_type": "test_lfm",
        "vocab_size": 32000,
        "hidden_size": 512,
        "num_layers": 4,
        "num_heads": 8,
        "total_parameters": total_params,
        "architecture": "simplified_lfm"
    }

    with open(f"{save_path}/config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Create tokenizer config (minimal)
    tokenizer_config = {
        "vocab_size": 32000,
        "model_max_length": 2048,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0
    }

    with open(f"{save_path}/tokenizer_config.json", 'w') as f:
        json.dump(tokenizer_config, f, indent=2)

    # Create generation config
    generation_config = {
        "max_length": 100,
        "temperature": 0.7,
        "do_sample": True,
        "pad_token_id": 0,
        "eos_token_id": 2
    }

    with open(f"{save_path}/generation_config.json", 'w') as f:
        json.dump(generation_config, f, indent=2)

    print("✅ Test LFM model created successfully!")
    print(f"📁 Model saved to: {save_path}")

    return model, sample_input

def convert_test_model_to_executorch(model_path="models/test_lfm"):
    """Convert the test model to ExecuTorch format"""

    print("🔄 Converting test model to ExecuTorch format...")

    # Load the test model with the same parameters that were saved
    model = TestLFMModel(
        vocab_size=32000,
        hidden_size=512,  # Match saved model
        num_layers=4,     # Match saved model
        num_heads=8       # Match saved model
    )
    model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin"))
    model.eval()

    # Create sample input
    sample_input = torch.randint(0, 32000, (1, 16))  # Smaller input for mobile

    print("📤 Exporting to Torch IR...")
    with torch.no_grad():
        exported = torch.export.export(model, (sample_input,))

    print("🔄 Converting to Edge format...")
    from executorch.exir import to_edge
    edge = to_edge(exported)

    # Apply SpaceGhost optimizations
    print("🧹 Applying SpaceGhost optimizations...")
    try:
        from research.spaceghost.patches.xnnpack.lfn_xnnpack_cleanup_pass import run_lfn_xnnpack_pipeline
        optimized_edge = run_lfn_xnnpack_pipeline(edge)
        spaceghost_applied = True
        print("✅ SpaceGhost optimizations applied")
    except ImportError:
        optimized_edge = edge
        spaceghost_applied = False
        print("⚠️  SpaceGhost not available, using standard pipeline")

    # Partition for XNNPack
    print("🎯 Partitioning for XNNPack...")
    from executorch.backends.xnnpack import XnnpackPartitioner
    partitioned = optimized_edge.to_backend(XnnpackPartitioner())

    # Count delegates
    delegate_count = sum(1 for node in partitioned.graph.nodes
                        if hasattr(node, 'target') and 'delegate' in str(node.target))
    print(f"✅ Created {delegate_count} delegate operations")

    # Convert to ExecuTorch
    print("⚡ Converting to ExecuTorch format...")
    executable = partitioned.to_executorch()

    # Save
    output_path = f"{model_path}/model.pte"
    print(f"💾 Saving to {output_path}...")
    with open(output_path, "wb") as f:
        executable.write_to_file(f)

    file_size = Path(output_path).stat().st_size
    print(".2f")

    # Create metadata
    metadata = {
        "model_name": "TestLFM-25M",
        "architecture": "simplified_lfm",
        "parameters": sum(p.numel() for p in model.parameters()),
        "spaceghost_optimized": spaceghost_applied,
        "delegate_operations": delegate_count,
        "pte_file_size": file_size,
        "sample_input_shape": list(sample_input.shape)
    }

    with open(f"{model_path}/model_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print("✅ Conversion complete!")
    return executable, metadata

def main():
    """Main function"""

    print("🚀 Test LFM Model Creator & Converter")
    print("=" * 45)

    # Create test model
    model, sample_input = create_test_lfm_model()

    # Convert to ExecuTorch
    executable, metadata = convert_test_model_to_executorch()

    print("\n" + "=" * 45)
    print("🎉 MODEL CREATION COMPLETE!")
    print("=" * 45)
    print(f"📁 Model: {metadata['model_name']}")
    print(",.1f")
    print(f"🎯 Delegates: {metadata['delegate_operations']}")
    print(".2f")
    print(f"🧹 SpaceGhost: {'✅ Applied' if metadata['spaceghost_optimized'] else '❌ Not applied'}")
    print()
    print("🚀 Ready for Motorola device testing!")
    print("   Deploy: ./scripts/deploy_device_spaceghost.sh")
    print("   Test: ./benchmark_lfm350_device.py")

if __name__ == "__main__":
    main()