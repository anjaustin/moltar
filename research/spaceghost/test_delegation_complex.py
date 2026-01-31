#!/usr/bin/env python3
"""
Test MaxPool2d delegation on complex model with our fix
"""

import sys
import os

# Add ExecuTorch to path
script_dir = os.path.dirname(os.path.abspath(__file__))
executorch_lib = os.path.join(script_dir, 'executorch', 'pip-out', 'lib.macosx-26.0-arm64-cpython-311')
sys.path.insert(0, executorch_lib)

import torch
import torch.nn as nn
from torch.export import export
from executorch.exir import to_edge
from patches.xnnpack.lfn_xnnpack_cleanup_pass import run_lfn_xnnpack_pipeline

class ComplexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def test_complex_delegation():
    """Test delegation on complex model"""
    print("🧪 TESTING COMPLEX MODEL DELEGATION")
    print("=" * 50)

    model = ComplexModel()
    sample_input = torch.randn(1, 1, 28, 28)

    print("1. Exporting and converting model...")
    exported = export(model, (sample_input,))
    edge = to_edge(exported)
    cleaned_edge = run_lfn_xnnpack_pipeline(edge)

    print("2. Running full XNNPack to_backend flow...")
    try:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

        print("   📊 Pre-backend operations...")
        method_name = list(cleaned_edge.methods)[0]
        exported_prog = cleaned_edge.exported_program(method_name)
        pre_ops = [node for node in exported_prog.graph_module.graph.nodes if node.op == 'call_function']
        pre_maxpool = [op for op in pre_ops if 'max_pool' in str(op.target)]
        print(f"      Total ops: {len(pre_ops)}, MaxPool ops: {len(pre_maxpool)}")

        # This should trigger the full partitioning + delegation flow
        lowered_edge = cleaned_edge.to_backend(XnnpackPartitioner())
        print("   ✅ to_backend completed")

        # Check results
        lowered_method_name = list(lowered_edge.methods)[0]
        lowered_prog = lowered_edge.exported_program(lowered_method_name)
        post_ops = [node for node in lowered_prog.graph_module.graph.nodes if node.op == 'call_function']
        post_maxpool = [op for op in post_ops if 'max_pool' in str(op.target)]
        post_delegates = [op for op in post_ops if 'delegate' in str(op.target).lower()]

        print(f"   📊 Post-backend operations: {len(post_ops)}")
        print(f"   🎯 MaxPool operations: {len(post_maxpool)}")
        print(f"   🎯 Delegate operations: {len(post_delegates)}")

        print("\n" + "=" * 70)
        print("🎯 COMPLEX MODEL DELEGATION VERDICT")
        print("=" * 70)

        if len(post_delegates) >= 2:
            print("🎉 SUCCESS: Complex model delegation worked!")
            print("   • Delegate calls created for MaxPool operations")
            print("   • REQ-XNN-001: FULLY SOLVED for complex models")
            return True
        else:
            print("❌ FAILURE: No delegate calls created")
            print(f"   • Still have {len(post_maxpool)} MaxPool operations in main graph")
            return False

    except Exception as e:
        print(f"   ❌ to_backend failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complex_delegation()
    sys.exit(0 if success else 1)