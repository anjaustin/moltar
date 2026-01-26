#!/usr/bin/env python3
"""
SpaceGhost: Detailed MaxPool2d XNNPack Analysis

Comprehensive analysis of MaxPool2d operations through the entire pipeline:
1. Original model operations
2. Exported operations
3. Edge operations
4. Partitioned operations
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

try:
    from executorch.exir import to_edge
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    EXECUTORCH_AVAILABLE = True
    print("✅ ExecuTorch modules loaded")
except ImportError as e:
    print(f"❌ ExecuTorch not available: {e}")
    EXECUTORCH_AVAILABLE = False
    exit(1)

class MaxPoolTestModel(nn.Module):
    """Simple model with MaxPool2d to test partitioning"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # The problematic operation
        self.fc = nn.Linear(64 * 16 * 16, 10)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)  # This should trigger MaxPool2d issues
        x = x.view(x.size(0), -1)
        return self.fc(x)

def analyze_operations(header, graph_module):
    """Analyze operations in a graph module"""
    print(f"\n🔍 {header}")
    print("-" * (len(header) + 3))

    operations = []
    for node in graph_module.graph.nodes:
        if node.op == 'call_function':
            op_name = str(node.target)
            operations.append(op_name)
            print(f"  📋 {op_name}")

            # Check for MaxPool2d related operations
            if 'max_pool' in op_name.lower():
                print(f"     🎯 MAXPOOL FOUND: {op_name}")
                print(f"     📝 Args: {node.args}")
                print(f"     📝 Kwargs: {node.kwargs}")

    maxpool_count = sum(1 for op in operations if 'max_pool' in op.lower())
    print(f"\n📊 Summary: {len(operations)} total operations, {maxpool_count} MaxPool operations")

    return operations

def main():
    """Main analysis function"""
    print("🔬 SpaceGhost: Detailed MaxPool2d XNNPack Analysis")
    print("=" * 55)

    # Create model and sample input
    model = MaxPoolTestModel()
    sample_input = torch.randn(1, 3, 32, 32)

    # Step 1: Analyze original model
    print("\n1️⃣ ORIGINAL MODEL ANALYSIS")
    print("torch.fx symbolic trace...")
    try:
        from torch.fx import symbolic_trace
        traced = symbolic_trace(model)
        analyze_operations("Original Model Operations", traced)
    except Exception as e:
        print(f"❌ Could not trace original model: {e}")

    # Step 2: Export and analyze
    print("\n2️⃣ TORCH.EXPORT ANALYSIS")
    exported = export(model, (sample_input,))
    analyze_operations("Exported Program Operations", exported.graph_module)

    # Step 3: Edge conversion and analyze
    print("\n3️⃣ EDGE CONVERSION ANALYSIS")
    edge = to_edge(exported)

    # Get the graph module from edge
    method_name = list(edge.methods)[0]
    exported_prog = edge.exported_program(method_name)
    edge_graph_module = exported_prog.graph_module
    analyze_operations("Edge Program Operations", edge_graph_module)

    # Step 4: Partitioning analysis
    print("\n4️⃣ XNNPACK PARTITIONING ANALYSIS")
    try:
        partitioner = XnnpackPartitioner()
        partitioned = partitioner(exported_prog)

        print("✅ Partitioning completed successfully")
        print(f"📦 Partition result type: {type(partitioned)}")

        # Analyze tagged exported program
        if hasattr(partitioned, 'tagged_exported_program'):
            tagged_ep = partitioned.tagged_exported_program
            print("📋 Tagged exported program available")
            analyze_operations("Partitioned Tagged Operations", tagged_ep.graph_module)

        # Analyze partition tags
        if hasattr(partitioned, 'partition_tags'):
            print(f"\n🏷️  Partition Tags: {len(partitioned.partition_tags)}")
            for tag, partition_info in partitioned.partition_tags.items():
                print(f"  🏷️  {tag}: {partition_info}")

                # Check if this partition contains MaxPool operations
                if hasattr(partition_info, 'graph_module'):
                    ops = analyze_operations(f"Partition '{tag}' Operations", partition_info.graph_module)
                    maxpool_in_partition = any('max_pool' in op.lower() for op in ops)
                    if maxpool_in_partition:
                        print(f"  🎯 MAXPOOL FOUND IN PARTITION '{tag}'!")
                        return True

        # Check if MaxPool2d is still in the main graph (not partitioned)
        if hasattr(partitioned, 'tagged_exported_program'):
            tagged_ops = []
            for node in partitioned.tagged_exported_program.graph_module.graph.nodes:
                if node.op == 'call_function':
                    tagged_ops.append(str(node.target))

            remaining_maxpool = [op for op in tagged_ops if 'max_pool' in op.lower()]
            if remaining_maxpool:
                print(f"\n⚠️  MaxPool operations still in main graph: {remaining_maxpool}")
                print("❌ FAILURE: MaxPool2d not successfully partitioned to XNNPack")
                return False
            else:
                print("\n✅ SUCCESS: No MaxPool operations remain in main graph")
                print("🎉 MaxPool2d successfully partitioned to XNNPack backend!")
                return True

    except Exception as e:
        print(f"❌ Partitioning failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()

    print("\n" + "=" * 55)
    print("📋 FINAL ANALYSIS RESULTS")
    print("=" * 55)

    if success:
        print("✅ REQ-XNN-001 SUCCESS: MaxPool2d partitioning working!")
        print("🎯 The MaxPool2dWithIndicesConfig fix is effective")
        print("🚀 Ready to proceed with Snapdragon 480 optimization")
    else:
        print("❌ REQ-XNN-001 FAILURE: MaxPool2d partitioning still broken")
        print("🔧 Need further investigation of the partitioning logic")
        print("📋 Check if MaxPool2dWithIndicesConfig is properly registered")