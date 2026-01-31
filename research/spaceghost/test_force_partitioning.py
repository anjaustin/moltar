#!/usr/bin/env python3
"""
SpaceGhost: Test Force Partitioning with Modified Config

Test if the regular XnnpackPartitioner works with our "force mode" MaxPool2dConfig.
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
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128 * 8 * 8, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        return self.fc(x.view(x.size(0), -1))

def test_force_partitioning():
    """Test force partitioning with modified config"""
    print("🧪 TESTING FORCE PARTITIONING WITH MODIFIED CONFIG")
    print("=" * 60)

    model = TestModel()
    sample_input = torch.randn(1, 3, 32, 32)

    print("1. Exporting and converting model...")
    exported = export(model, (sample_input,))
    edge = to_edge(exported)

    print("2. Applying LFN cleanup pipeline...")
    cleaned_edge = run_lfn_xnnpack_pipeline(edge)

    print("3. Checking pre-partitioning state...")
    method_name = list(cleaned_edge.methods)[0]
    exported_prog = cleaned_edge.exported_program(method_name)
    pre_ops = [node for node in exported_prog.graph_module.graph.nodes if node.op == 'call_function']
    pre_maxpool = [op for op in pre_ops if 'max_pool' in str(op.target)]
    print(f"   📊 Pre-partition: {len(pre_ops)} total ops, {len(pre_maxpool)} MaxPool ops")

    print("\n4. APPLYING REGULAR PARTITIONER (with force config)...")
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

    partitioner = XnnpackPartitioner()
    partitioned = partitioner(exported_prog)

    print("5. Checking post-partitioning state...")
    if hasattr(partitioned, 'tagged_exported_program'):
        tagged_ep = partitioned.tagged_exported_program
        post_ops = [node for node in tagged_ep.graph_module.graph.nodes if node.op == 'call_function']
        post_maxpool = [op for op in post_ops if 'max_pool' in str(op.target)]
        delegate_calls = [op for op in post_ops if 'delegate' in str(op.target).lower() or 'call_' in str(op.target).lower()]

        print(f"   📊 Post-partition: {len(post_ops)} total ops, {len(post_maxpool)} MaxPool ops")
        print(f"   🎯 Delegate calls: {len(delegate_calls)}")

        if delegate_calls:
            print("   🎯 Delegate nodes found:")
            for call in delegate_calls:
                print(f"      • {call.name}: {call.target}")

        print("\n" + "=" * 60)
        print("🎯 FORCE PARTITIONING VERDICT")
        print("=" * 60)

        if len(post_maxpool) < len(pre_maxpool):
            print("🎉 SUCCESS: Force partitioning worked!")
            print(f"   • MaxPool ops reduced: {len(pre_maxpool)} → {len(post_maxpool)}")
            print("   • REQ-XNN-001: SOLVED via force config")
            if delegate_calls:
                print("   • Delegate nodes created - operations in XNNPack subgraphs")
            return True
        else:
            print("❌ FAILURE: Force partitioning did not reduce MaxPool operations")
            print(f"   • MaxPool ops unchanged: {len(pre_maxpool)} → {len(post_maxpool)}")
            print("   • The framework bug persists")
            return False
    else:
        print("❌ No tagged_exported_program returned")
        return False

if __name__ == "__main__":
    success = test_force_partitioning()
    sys.exit(0 if success else 1)