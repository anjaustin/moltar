#!/usr/bin/env python3
"""
Test the Manual Delegation Fix for ExecuTorch Ghost Partition Bug
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

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.pool(x)

def test_manual_delegation_fix():
    """Test the manual delegation fix"""
    print("🧪 TESTING MANUAL DELEGATION FIX")
    print("=" * 50)

    model = TestModel()
    sample_input = torch.randn(1, 3, 32, 32)

    print("1. Exporting and converting model...")
    exported = export(model, (sample_input,))
    edge = to_edge(exported)
    cleaned_edge = run_lfn_xnnpack_pipeline(edge)

    print("2. Applying manual delegation fix...")
    from manual_delegation_fix import apply_manual_delegation_fix

    try:
        fixed_edge = apply_manual_delegation_fix(cleaned_edge)
        print("   ✅ Manual delegation fix applied")
    except Exception as e:
        print(f"   ❌ Manual delegation fix failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n3. Analyzing results...")
    method_name = list(fixed_edge.methods)[0]
    exported_prog = fixed_edge.exported_program(method_name)

    # Count operations
    ops = [node for node in exported_prog.graph_module.graph.nodes if node.op == 'call_function']
    maxpool_ops = [op for op in ops if 'max_pool' in str(op.target)]
    delegate_ops = [op for op in ops if 'delegate' in str(op.target).lower()]

    print(f"   📊 Total operations: {len(ops)}")
    print(f"   🎯 MaxPool operations: {len(maxpool_ops)}")
    print(f"   🎯 Delegate operations: {len(delegate_ops)}")

    if delegate_ops:
        print("   🎯 Delegate operations found:")
        for delegate in delegate_ops:
            print(f"      • {delegate.name}: {delegate.target}")

    print("\n" + "=" * 70)
    print("🎯 MANUAL DELEGATION FIX VERDICT")
    print("=" * 70)

    if len(delegate_ops) > 0:
        print("🎉 SUCCESS: Manual delegation fix worked!")
        print("   • Delegate calls created - operations are in XNNPack subgraphs")
        print("   • ExecuTorch 'Ghost Partition' bug has been bypassed")
        print("   • REQ-XNN-001: SOLVED via manual delegation")
        return True
    else:
        print("❌ FAILURE: Manual delegation fix did not create delegate calls")
        print("   • The fix needs further debugging")
        return False

if __name__ == "__main__":
    success = test_manual_delegation_fix()
    sys.exit(0 if success else 1)