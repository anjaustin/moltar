#!/usr/bin/env python3
"""
SpaceGhost: MaxPool2d XNNPack Debugging Script

This script creates a test model with MaxPool2d operations to diagnose
the XNNPack backend delegation issues identified in our research.

Based on web research findings:
- XNNPack fails on MaxPool2d due to partitioner issues
- MaxPool2d with indices output not properly handled
- NHWC memory format requirements not met

Usage:
    python debug_maxpool2d.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import export
import json
import os
from pathlib import Path

# SpaceGhost imports
import sys
import os

# Add ExecuTorch to path
script_dir = os.path.dirname(os.path.abspath(__file__))
executorch_lib = os.path.join(script_dir, 'executorch', 'pip-out', 'lib.macosx-26.0-arm64-cpython-311')
sys.path.insert(0, executorch_lib)

try:
    from executorch.exir import to_edge
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.devtools.inspector import Inspector
    from executorch.backends.xnnpack import XnnpackBackend
    EXECUTORCH_AVAILABLE = True
    print("✅ ExecuTorch modules loaded successfully")
except ImportError as e:
    print(f"ExecuTorch not available: {e}")
    EXECUTORCH_AVAILABLE = False

class TestModelWithMaxPool2d(nn.Module):
    """Simple CNN model that includes MaxPool2d operations"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((4, 4))
        self.fc = nn.Linear(128 * 4 * 4, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # This should trigger MaxPool2d issues
        x = F.relu(self.conv2(x))
        x = self.adaptive_pool(x)  # Another pooling operation
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class LiquidInspiredModel(nn.Module):
    """Model inspired by Liquid Foundation Networks with MaxPool2d"""

    def __init__(self):
        super().__init__()
        # Simplified LFN-like architecture
        self.input_proj = nn.Linear(784, 256)
        self.liquid_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        # Add conv layers that might use MaxPool2d
        self.conv_proj = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(2, 2)  # This is the problematic operation
        self.output_proj = nn.Linear(64 + 32*14*14, 10)  # 14x14 after pooling 28x28->14x14

    def forward(self, x):
        # Flatten for initial projection
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)

        # Liquid processing
        liquid_out = self.liquid_layer(self.input_proj(x_flat))

        # Convolutional processing with MaxPool2d
        x_conv = x.view(batch_size, 1, 28, 28)  # MNIST-like input
        x_conv = F.relu(self.conv_proj(x_conv))
        x_conv = self.maxpool(x_conv)  # This should fail in XNNPack
        x_conv = x_conv.view(batch_size, -1)

        # Combine and output
        combined = torch.cat([liquid_out, x_conv], dim=1)
        return self.output_proj(combined)

def create_test_models():
    """Create test models for MaxPool2d debugging"""
    models = {
        'cnn_maxpool': TestModelWithMaxPool2d(),
        'liquid_maxpool': LiquidInspiredModel()
    }
    return models

def export_model_to_executorch(model, model_name, sample_input):
    """Export model to ExecuTorch format"""
    print(f"\n=== Exporting {model_name} ===")

    try:
        # Export to torch.export format
        exported_program = export(model, (sample_input,))
        print(f"✓ Torch export successful for {model_name}")

        # Convert to edge format
        edge_program = to_edge(exported_program)
        print(f"✓ Edge conversion successful for {model_name}")

        return edge_program

    except Exception as e:
        print(f"✗ Export failed for {model_name}: {e}")
        return None

def analyze_with_inspector(edge_program, model_name):
    """Analyze the exported program with ExecuTorch Inspector"""
    print(f"\n=== Inspector Analysis: {model_name} ===")

    try:
        # Get the graph_module from the edge program
        method_name = list(edge_program.methods)[0]
        graph_module = edge_program.exported_program(method_name).graph_module

        inspector = Inspector(graph_module)
        inspector.print_summary()

        # Get detailed node information
        print("\n--- Node Details ---")
        for node in graph_module.graph.nodes:
            if 'max_pool' in str(node).lower() or 'pool' in str(node).lower():
                print(f"Found pooling node: {node}")
                print(f"  Op: {node.op}")
                print(f"  Target: {node.target}")
                print(f"  Args: {node.args}")
                print(f"  Kwargs: {node.kwargs}")

        return inspector

    except Exception as e:
        print(f"✗ Inspector analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_xnnpack_partitioning(edge_program, model_name):
    """Test XNNPack partitioning on the edge program"""
    print(f"\n=== XNNPack Partitioning Test: {model_name} ===")

    try:
        # Create XNNPack partitioner
        partitioner = XnnpackPartitioner()

        # Get the graph_module from the edge program
        method_name = list(edge_program.methods)[0]  # Get first method name
        exported_prog = edge_program.exported_program(method_name)

        # Try partitioning on the ExportedProgram directly
        try:
            partitioned_program = partitioner(exported_prog)
            print(f"✓ XNNPack partitioning successful for {model_name} (ExportedProgram)")

            # Check for delegated nodes - need to inspect the result
            print(f"Partitioned result type: {type(partitioned_program)}")
            return partitioned_program

        except Exception as e1:
            print(f"ExportedProgram partitioning failed, trying graph_module: {e1}")

            # Fallback: try with graph_module
            graph_module = exported_prog.graph_module
            partitioned_program = partitioner(graph_module)
            print(f"✓ XNNPack partitioning successful for {model_name} (graph_module)")

            # Check for delegated nodes
            delegated_count = 0
            for node in partitioned_program.graph.nodes:
                if hasattr(node, 'meta') and 'delegate' in str(node.meta):
                    delegated_count += 1
                    print(f"Delegated node: {node}")

            print(f"Total delegated nodes: {delegated_count}")
            return partitioned_program

    except Exception as e:
        print(f"✗ XNNPack partitioning failed for {model_name}: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None

def test_xnnpack_lowering(partitioned_program, model_name):
    """Test lowering partitioned program to XNNPack"""
    print(f"\n=== XNNPack Lowering Test: {model_name} ===")

    try:
        # Create XNNPack backend
        backend = XnnpackBackend()

        # Attempt lowering
        lowered_program = backend.compile(partitioned_program)
        print(f"✓ XNNPack lowering successful for {model_name}")

        return lowered_program

    except Exception as e:
        print(f"✗ XNNPack lowering failed for {model_name}: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None

def diagnose_maxpool2d_issues():
    """Main diagnostic function for MaxPool2d issues"""
    print("🚀 SpaceGhost: MaxPool2d XNNPack Diagnostic Tool")
    print("=" * 50)

    if not EXECUTORCH_AVAILABLE:
        print("❌ ExecuTorch not available. Please run from SpaceGhost environment.")
        return

    # Create test models
    models = create_test_models()

    # Sample inputs
    sample_inputs = {
        'cnn_maxpool': torch.randn(1, 3, 32, 32),  # CIFAR-like
        'liquid_maxpool': torch.randn(1, 784)  # MNIST-like flattened
    }

    results = {}

    for model_name, model in models.items():
        print(f"\n🔍 Analyzing {model_name}...")
        sample_input = sample_inputs[model_name]

        # Step 1: Export to ExecuTorch
        edge_program = export_model_to_executorch(model, model_name, sample_input)
        if not edge_program:
            continue

        # Step 2: Inspector analysis
        inspector = analyze_with_inspector(edge_program, model_name)

        # Step 3: Test XNNPack partitioning
        partitioned = test_xnnpack_partitioning(edge_program, model_name)
        if not partitioned:
            results[model_name] = "Partitioning Failed"
            continue

        # Step 4: Test XNNPack lowering
        lowered = test_xnnpack_lowering(partitioned, model_name)
        if lowered:
            results[model_name] = "Success"
        else:
            results[model_name] = "Lowering Failed"

    # Summary
    print("\n" + "=" * 50)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 50)

    for model_name, status in results.items():
        status_icon = "✅" if status == "Success" else "❌"
        print(f"{status_icon} {model_name}: {status}")

    # Analysis
    print("\n🔬 ANALYSIS:")
    if all(status == "Success" for status in results.values()):
        print("✅ All models processed successfully - MaxPool2d may not be the issue")
    else:
        print("❌ Issues detected - MaxPool2d problems confirmed")
        print("\n💡 RECOMMENDATIONS:")
        print("1. Check if MaxPool2d nodes are being partitioned")
        print("2. Verify NHWC memory format requirements")
        print("3. Implement custom MaxPool2d handler in XnnpackPartitioner")
        print("4. Add indices output handling for aten.max_pool2d_with_indices")

def main():
    """Main execution"""
    diagnose_maxpool2d_issues()

if __name__ == "__main__":
    main()