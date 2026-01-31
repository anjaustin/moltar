#!/usr/bin/env python3
"""
Convert LFM2-700M to ExecuTorch format using SpaceGhost venv
"""

import os
import sys
import subprocess

def convert_with_spaceghost_venv():
    """Convert using the SpaceGhost virtual environment"""

    # Activate SpaceGhost venv and run conversion
    cmd = """
cd /Users/aaronjosserand-austin/000/Motorola/research/brack
source ../spaceghost/venv/bin/activate

python3 -c "
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from executorch.exir import EdgeProgramManager, ExecutorchProgramManager
from executorch.backends.xnnpack import XnnpackBackend
import json

print('🚀 Converting LFM2-700M to ExecuTorch (.pte) format...')

# Load model
model_path = 'models/LFM2-700M'
print(f'📦 Loading model from: {model_path}')
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16,
    device_map='cpu',
    low_cpu_mem_usage=True
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print('🎯 Preparing for ExecuTorch export...')

# Export the model itself (simpler approach)
print('⚡ Exporting LFM2-700M model to ExecuTorch with XNNPack backend...')
print('Using simplified forward pass export...')

# Create simple input for export
example_input = torch.randint(0, tokenizer.vocab_size, (1, 10))

try:
    # Export the model directly - simpler approach
    edge_program = torch.export.export(
        model,
        (example_input,),
        strict=False
    )

    edge_manager = EdgeProgramManager(edge_program)
    edge_manager = edge_manager.to_backend(XnnpackBackend())

    executorch_program = ExecutorchProgramManager(edge_manager).to_executorch()

    # Save
    output_path = 'models/LFM2-700M/model.pte'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        executorch_program.write_to_file(f)

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f'💾 Saved ExecuTorch model: {output_path}')
    print(f'📊 Model size: {file_size:.1f}MB')

    print('✅ Conversion successful!')
    print('🚀 Ready for Motorola deployment with ExecuTorch runtime!')

except Exception as e:
    print(f'❌ Conversion failed: {e}')
    import traceback
    traceback.print_exc()
"
"""

    print("🔄 Converting LFM2-700M to ExecuTorch format...")
    print("Using SpaceGhost venv with ExecuTorch...")

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("❌ Conversion timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

def check_conversion_result():
    """Check if conversion was successful"""
    pte_path = "models/LFM2-700M/model.pte"
    if os.path.exists(pte_path):
        size = os.path.getsize(pte_path) / (1024 * 1024)
        print(f"✅ ExecuTorch model created: {size:.1f}MB")
        return True
    else:
        print("❌ No .pte file found")
        return False

if __name__ == "__main__":
    print("🚀 LFM2-700M ExecuTorch Conversion (Real)")
    print("=" * 45)

    success = convert_with_spaceghost_venv()

    if success:
        print("\n🎉 CONVERSION ATTEMPTED!")
        check_conversion_result()
    else:
        print("\n❌ Conversion failed")

    print("\n🔍 Check the output above for details...")