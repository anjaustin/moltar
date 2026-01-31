#!/usr/bin/env python3
"""
Debug XNNPack partitioner by creating a custom version with logging.
"""

import sys
import os
import torch

# Add ExecuTorch to path
script_dir = os.path.dirname(os.path.abspath(__file__))
executorch_lib = os.path.join(script_dir, 'executorch', 'pip-out', 'lib.macosx-26.0-arm64-cpython-311')
sys.path.insert(0, executorch_lib)

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir.backend.partitioner import PartitionResult
from executorch.exir.backend.canonical_partitioners.config_partitioner import format_target_name

class DebugXnnpackPartitioner(XnnpackPartitioner):
    """XNNPack partitioner with debug logging"""

    def partition(self, exported_program):
        print("🔧 DEBUG PARTITIONER: Starting partition process")

        # Call parent's generate_partitions method
        partitions = self.generate_partitions(exported_program)
        print(f"🔧 DEBUG PARTITIONER: Generated {len(partitions)} partitions")

        for i, partition in enumerate(partitions):
            print(f"  Partition {i}: {len(partition.nodes)} nodes")
            for node in partition.nodes:
                print(f"    • {node.name}: {node.target}")

        # Call parent's partition method
        result = super().partition(exported_program)

        print(f"🔧 DEBUG PARTITIONER: Final result has {len(result.partition_tags)} tags")
        for tag, spec in result.partition_tags.items():
            print(f"  Tag {tag}: {spec.backend_id}")

        # Check tagged nodes
        tagged_count = 0
        for node in exported_program.graph_module.graph.nodes:
            if hasattr(node, 'meta') and 'delegation_tag' in node.meta:
                tagged_count += 1
                print(f"  Tagged node: {node.name} -> {node.meta['delegation_tag']}")

        print(f"🔧 DEBUG PARTITIONER: Total tagged nodes: {tagged_count}")

        return result

def test_debug_partitioner():
    """Test with debug partitioner"""
    print("🧪 TESTING WITH DEBUG PARTITIONER")
    print("=" * 50)

    from torch.export import export
    from executorch.exir import to_edge
    from patches.xnnpack.lfn_xnnpack_cleanup_pass import run_lfn_xnnpack_pipeline

    # Create test model
    import torch.nn as nn
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

    model = TestModel()
    sample_input = torch.randn(1, 3, 32, 32)

    print("1. Exporting and converting model...")
    exported = export(model, (sample_input,))
    edge = to_edge(exported)

    print("2. Applying LFN cleanup pipeline...")
    cleaned_edge = run_lfn_xnnpack_pipeline(edge)

    print("3. Applying DEBUG XNNPack partitioner...")
    debug_partitioner = DebugXnnpackPartitioner()
    method_name = list(cleaned_edge.methods)[0]
    exported_prog = cleaned_edge.exported_program(method_name)

    partitioned = debug_partitioner.partition(exported_prog)

    print("\n4. Final analysis...")
    if hasattr(partitioned, 'tagged_exported_program'):
        tagged_ep = partitioned.tagged_exported_program
        final_ops = [node for node in tagged_ep.graph_module.graph.nodes if node.op == 'call_function']
        final_maxpool = [op for op in final_ops if 'max_pool' in str(op.target)]
        final_delegates = [op for op in final_ops if 'delegate' in str(op.target).lower()]

        print(f"   📊 Final operations: {len(final_ops)}")
        print(f"   🎯 MaxPool operations: {len(final_maxpool)}")
        print(f"   🎯 Delegate operations: {len(final_delegates)}")

        success = len(final_maxpool) == 0 and len(final_delegates) > 0
        if success:
            print("   ✅ SUCCESS: MaxPool operations successfully delegated!")
        else:
            print("   ❌ FAILURE: Delegation did not occur")

        return success
    else:
        print("   ❌ No tagged_exported_program")
        return False

if __name__ == "__main__":
    success = test_debug_partitioner()
    sys.exit(0 if success else 1)