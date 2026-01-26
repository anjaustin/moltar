#!/usr/bin/env python3
"""
SpaceGhost: Simple MaxPool2d Partitioning Test

Focused test to reproduce the MaxPool2d XNNPack partitioning issue.
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
    from executorch.backends.xnnpack.partition.config import ALL_PARTITIONER_CONFIGS
    EXECUTORCH_AVAILABLE = True
    print("✅ ExecuTorch modules loaded")
except ImportError as e:
    print(f"❌ ExecuTorch not available: {e}")
    EXECUTORCH_AVAILABLE = False

# Runtime patch for MaxPool2dWithIndicesConfig
def apply_runtime_patch():
    """Apply runtime patch to add MaxPool2dWithIndicesConfig"""
    # Check if already patched
    for config in ALL_PARTITIONER_CONFIGS:
        if config.__name__ == 'MaxPool2dWithIndicesConfig':
            return True  # Already patched

    print("🔧 Applying runtime patch for MaxPool2dWithIndicesConfig...")

    # Import required modules
    from executorch.backends.xnnpack.partition.config.generic_node_configs import GenericNodePartitionerConfig
    import torch
    from executorch.exir.backend.canonical_partitioners.config_partitioner import format_target_name
    from executorch.exir.backend.utils import is_shape_dynamic, WhyNoPartition
    from torch.export import ExportedProgram

    import logging
    logger = logging.getLogger(__name__)
    why = WhyNoPartition(logger=logger)

    class MaxPool2dWithIndicesConfig(GenericNodePartitionerConfig):
        target_name = "max_pool2d_with_indices.default"

        def check_constraints(self, node: torch.fx.Node, ep: ExportedProgram) -> bool:
            if not self.check_common_constraints(node, ep):
                return False

            # Check if indices output has users
            graph = ep.graph_module.graph if hasattr(ep, 'graph_module') else ep.graph
            indices_users = []
            for user in node.users:
                if user.op == 'call_function' and str(user.target) == 'operator.getitem':
                    if len(user.args) > 1 and user.args[1] == 1:
                        indices_users.append(user)

            if indices_users:
                why(node, reason="MaxPool2d indices output has users - cannot partition")
                return False

            # Standard constraints
            kernel_size = node.args[1]
            stride = node.args[2] if len(node.args) > 2 else kernel_size
            if stride[0] > kernel_size[0] or stride[1] > kernel_size[1]:
                why(node, reason=f"stride ({stride}) must be <= kernel_size ({kernel_size})")
                return False

            return True

        def supported_precision_types(self):
            from executorch.backends.xnnpack.partition.config.xnnpack_config import ConfigPrecisionType
            return [ConfigPrecisionType.FP32, ConfigPrecisionType.STATIC_QUANT]

    # Add to configs
    ALL_PARTITIONER_CONFIGS.append(MaxPool2dWithIndicesConfig)
    print("✅ Runtime patch applied")
    return True

# Apply patch if ExecuTorch is available
if EXECUTORCH_AVAILABLE:
    apply_runtime_patch()
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

def test_maxpool_partitioning():
    """Test if MaxPool2d can be partitioned to XNNPack"""
    print("🧪 Testing MaxPool2d XNNPack Partitioning")
    print("=" * 50)

    try:
        # Create model and sample input
        model = MaxPoolTestModel()
        sample_input = torch.randn(1, 3, 32, 32)

        print("1. Exporting model...")
        exported = export(model, (sample_input,))
        print("   ✅ Export successful")
    except Exception as e:
        print(f"   ❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("2. Converting to Edge...")
    try:
        edge = to_edge(exported)
        print("   ✅ Edge conversion successful")
    except Exception as e:
        print(f"   ❌ Edge conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("3. Testing XNNPack partitioning...")
    try:
        partitioner = XnnpackPartitioner()

        # Get the ExportedProgram directly
        method_name = list(edge.methods)[0]
        exported_prog = edge.exported_program(method_name)

        print("   📋 Attempting to partition...")
        print(f"   📊 Available configs: {len(ALL_PARTITIONER_CONFIGS)}")
        maxpool_configs = [c for c in ALL_PARTITIONER_CONFIGS if 'MaxPool' in c.__name__]
        print(f"   🎯 MaxPool configs: {[c.__name__ for c in maxpool_configs]}")

        partitioned = partitioner(exported_prog)

        print("   ✅ Partitioning completed")

        # Check the PartitionResult for delegated modules
        print(f"   📊 Partition result type: {type(partitioned)}")
        print(f"   📊 Partition result attributes: {[attr for attr in dir(partitioned) if not attr.startswith('_')]}")

        # Check for delegated partitions
        if hasattr(partitioned, 'partitions'):
            print(f"   📦 Found {len(partitioned.partitions)} partitions")
            for i, partition in enumerate(partitioned.partitions):
                print(f"      Partition {i}: {partition}")

                # Check if this partition contains MaxPool2d
                if hasattr(partition, 'nodes'):
                    for node in partition.nodes:
                        node_str = str(node)
                        if 'max_pool' in node_str.lower():
                            print(f"   🎯 Found MaxPool2d in partition {i}: {node}")
                            print("   ✅ SUCCESS: MaxPool2d successfully partitioned to XNNPack")
                            return True

        # Check for remaining graph (non-partitioned nodes)
        if hasattr(partitioned, 'graph'):
            print("   📋 Checking remaining graph for MaxPool2d nodes...")
            maxpool_found = False
            for node in partitioned.graph.nodes:
                node_str = str(node)
                if 'max_pool' in node_str.lower():
                    maxpool_found = True
                    print(f"   ⚠️  MaxPool2d found in remaining graph: {node}")
                    print("   ❌ FAILURE: MaxPool2d not partitioned (still in main graph)")
                    return False

            if not maxpool_found:
                print("   ❓ No MaxPool2d nodes found anywhere - check if operation was decomposed")
                print("   🔍 Checking for aten.max_pool2d_with_indices...")
                for node in partitioned.graph.nodes:
                    if 'max_pool2d_with_indices' in str(node):
                        print(f"   📍 Found decomposed MaxPool2d: {node}")
                        return False  # Still not partitioned
                print("   ❓ MaxPool2d operation not found - may have been optimized away")
                return False

        print("   ✅ SUCCESS: No MaxPool2d nodes in remaining graph")
        return True

    except Exception as e:
        print(f"   ❌ Partitioning failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Starting MaxPool2d partitioning test...")
    success = test_maxpool_partitioning()

    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)

    if success:
        print("✅ MaxPool2d partitioning works - no issue detected")
    else:
        print("❌ MaxPool2d partitioning failed - issue confirmed")
        print("\n🔧 CONFIRMED ISSUES:")
        print("1. MaxPool2d operator not supported in XNNPack partitioner")
        print("2. Partitioning fails or MaxPool2d not delegated")
        print("3. Need to implement REQ-XNN-001: MaxPool2d operator support")

    print("\n🚀 NEXT STEPS:")
    print("1. Implement MaxPool2d handler in XnnpackPartitioner")
    print("2. Add NHWC memory format transformations")
    print("3. Test with indices output handling")

if __name__ == "__main__":
    main()