#!/usr/bin/env python3
"""
Convert LiquidAI LFM model to ExecuTorch format for mobile deployment
"""

import os
import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def convert_lfm_to_executorch(model_path: str, output_path: str):
    """
    Convert LiquidAI LFM model to ExecuTorch format

    Args:
        model_path: Path to the downloaded model directory
        output_path: Path where to save the .pte file
    """
    print(f"🔄 Converting LFM model to ExecuTorch format...")
    print(f"   Model: {model_path}")
    print(f"   Output: {output_path}")

    # Load model and tokenizer
    print("📥 Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # Use float32 for mobile compatibility
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)

    print(f"✅ Model loaded: {config.model_type}")
    print(f"   Parameters: ~{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"   Vocab size: {tokenizer.vocab_size}")

    # Set model to evaluation mode
    model.eval()

    # Create sample input for export
    print("🎯 Creating sample input for export...")
    sample_text = "Hello, how are you?"
    inputs = tokenizer(sample_text, return_tensors="pt")

    # Get the actual input that the model expects
    # For causal LMs, we typically use input_ids
    sample_input = inputs["input_ids"]

    print(f"   Sample input shape: {sample_input.shape}")
    print(f"   Sample input: {sample_text}")

    # Apply SpaceGhost optimizations before export
    print("🧹 Applying SpaceGhost optimizations...")

    # Import SpaceGhost cleanup pass
    try:
        from research.spaceghost.patches.xnnpack.lfn_xnnpack_cleanup_pass import run_lfn_xnnpack_pipeline
        print("✅ SpaceGhost cleanup pass available")
    except ImportError:
        print("⚠️  SpaceGhost cleanup pass not available - using standard export")
        run_lfn_xnnpack_pipeline = None

    # Export to Torch IR
    print("📤 Exporting to Torch IR...")
    with torch.no_grad():
        exported = torch.export.export(model, (sample_input,))

    # Convert to Edge format
    print("🔄 Converting to Edge format...")
    from executorch.exir import to_edge
    edge = to_edge(exported)

    # Apply SpaceGhost optimizations if available
    if run_lfn_xnnpack_pipeline:
        print("🧹 Running SpaceGhost cleanup pass...")
        edge = run_lfn_xnnpack_pipeline(edge)
        print("✅ SpaceGhost optimizations applied")
    else:
        print("⚠️  Skipping SpaceGhost optimizations")

    # Partition for XNNPack
    print("🎯 Partitioning for XNNPack...")
    from executorch.backends.xnnpack import XnnpackPartitioner
    partitioned = edge.to_backend(XnnpackPartitioner())

    # Count delegate operations
    delegate_count = 0
    for node in partitioned.graph.nodes:
        if hasattr(node, 'target') and 'delegate' in str(node.target):
            delegate_count += 1

    print(f"✅ Partitioned with {delegate_count} delegate operations")

    # Convert to ExecuTorch
    print("⚡ Converting to ExecuTorch format...")
    executable = partitioned.to_executorch()

    # Save the model
    print(f"💾 Saving ExecuTorch model to {output_path}...")
    with open(output_path, "wb") as f:
        executable.write_to_file(f)

    # Verify the saved model
    file_size = os.path.getsize(output_path)
    print(".2f")
    print("✅ Model conversion complete!")

    # Create metadata
    metadata = {
        "model_name": config.name_or_path,
        "model_type": config.model_type,
        "parameters": sum(p.numel() for p in model.parameters()),
        "vocab_size": tokenizer.vocab_size,
        "max_position_embeddings": getattr(config, 'max_position_embeddings', 'unknown'),
        "torch_version": torch.__version__,
        "sample_input_shape": list(sample_input.shape),
        "sample_input_text": sample_text,
        "spaceghost_optimized": run_lfn_xnnpack_pipeline is not None,
        "delegate_operations": delegate_count,
        "pte_file_size": file_size,
        "conversion_timestamp": str(torch.randint(0, 1000000, (1,)).item())  # Placeholder
    }

    # Save metadata
    metadata_path = output_path.replace('.pte', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"📋 Metadata saved to {metadata_path}")

    return executable, metadata

def main():
    """Main conversion function"""

    # Model paths
    model_name = "LFM2.5-1.2B-Instruct"
    model_dir = f"models/{model_name}"
    output_pte = f"{model_dir}/model.pte"

    print("🚀 LFM to ExecuTorch Converter")
    print("=" * 40)
    print(f"Model: {model_name}")
    print(f"Source: {model_dir}")
    print(f"Output: {output_pte}")
    print()

    # Check if model exists
    if not os.path.exists(model_dir):
        print(f"❌ Model directory not found: {model_dir}")
        return False

    if not os.path.exists(f"{model_dir}/model.safetensors"):
        print(f"❌ Model file not found: {model_dir}/model.safetensors")
        return False

    # Check if output already exists
    if os.path.exists(output_pte):
        response = input(f"Output file {output_pte} already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Conversion cancelled.")
            return False

    try:
        # Convert the model
        executable, metadata = convert_lfm_to_executorch(model_dir, output_pte)

        print("\n" + "=" * 40)
        print("🎉 CONVERSION SUCCESSFUL!")
        print("=" * 40)
        print(f"📁 Model: {metadata['model_name']}")
        print(",.1f")
        print(f"📝 Vocab: {metadata['vocab_size']}")
        print(f"🎯 Delegates: {metadata['delegate_operations']}")
        print(".2f")
        print(f"🧹 SpaceGhost: {'✅ Applied' if metadata['spaceghost_optimized'] else '❌ Not applied'}")
        print()
        print("🚀 Ready for mobile deployment!")
        print(f"   Deploy: ./scripts/deploy_device_spaceghost.sh")
        print(f"   Test: ./benchmark_lfm350_device.py (update paths)")

        return True

    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)