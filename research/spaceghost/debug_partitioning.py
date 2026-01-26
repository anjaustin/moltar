#!/usr/bin/env python3
"""
SpaceGhost: Partitioning Debug Script

Debug the XNNPack partitioning process to see why MaxPool2dWithIndicesConfig isn't matching.
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
    exit(1)

class MaxPoolTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 16 * 16, 10)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def debug_config_loading():
    """Debug which configs are loaded"""
    print("🔍 Checking loaded partitioner configs...")
    print(f"📊 Total configs: {len(ALL_PARTITIONER_CONFIGS)}")

    maxpool_configs = [config for config in ALL_PARTITIONER_CONFIGS
                      if 'MaxPool' in config.__name__]

    print(f"🎯 MaxPool configs found: {len(maxpool_configs)}")
    for config in maxpool_configs:
        print(f"   • {config.__name__}: target_name = {config.target_name}")

    # Check if our config is there
    our_config = None
    for config in ALL_PARTITIONER_CONFIGS:
        if config.__name__ == 'MaxPool2dWithIndicesConfig':
            our_config = config
            break

    if our_config:
        print("✅ MaxPool2dWithIndicesConfig found in ALL_PARTITIONER_CONFIGS")
        print(f"   Target: {our_config.target_name}")
        print(f"   Supported precisions: {our_config().supported_precision_types()}")
    else:
        print("❌ MaxPool2dWithIndicesConfig NOT found in ALL_PARTITIONER_CONFIGS")
        print("🔧 Applying runtime patch...")

        # Runtime patch: Add our config to the list
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

        # Add to the configs list
        ALL_PARTITIONER_CONFIGS.append(MaxPool2dWithIndicesConfig)
        print("✅ Runtime patch applied - MaxPool2dWithIndicesConfig added")

        return MaxPool2dWithIndicesConfig

    return our_config

def debug_target_matching():
    """Debug the target name matching process"""
    print("\n🔍 Testing target name matching...")

    # Create test model and get edge program
    model = MaxPoolTestModel()
    sample_input = torch.randn(1, 3, 32, 32)

    exported = export(model, (sample_input,))
    edge = to_edge(exported)

    method_name = list(edge.methods)[0]
    exported_prog = edge.exported_program(method_name)
    graph_module = exported_prog.graph_module

    print("📋 Operations in graph:")
    maxpool_nodes = []
    for node in graph_module.graph.nodes:
        if node.op == 'call_function':
            op_name = str(node.target)
            print(f"   • {op_name}")

            if 'max_pool' in op_name.lower():
                maxpool_nodes.append(node)
                print(f"     🎯 MAXPOOL NODE: {node}")
                print(f"     📝 node.target: {node.target}")
                print(f"     📝 node.target.__name__: {node.target.__name__}")

    # Test format_target_name function
    try:
        from executorch.exir.backend.canonical_partitioners.config_partitioner import format_target_name

        for node in maxpool_nodes:
            formatted = format_target_name(node.target.__name__)
            print(f"     🔄 Formatted target: '{formatted}'")

            # Check if our config matches
            our_config = None
            for config in ALL_PARTITIONER_CONFIGS:
                if hasattr(config, 'target_name') and config.target_name == formatted:
                    our_config = config
                    print(f"     ✅ MATCH: {config.__name__} matches '{formatted}'")
                    break

            if not our_config:
                print(f"     ❌ NO MATCH: No config found for '{formatted}'")
                # Show what configs we do have
                similar_configs = [c for c in ALL_PARTITIONER_CONFIGS
                                 if hasattr(c, 'target_name') and 'max_pool' in c.target_name]
                if similar_configs:
                    print(f"     💡 Similar configs: {[c.__name__ for c in similar_configs]}")

    except ImportError:
        print("❌ Could not import format_target_name function")

    return maxpool_nodes

def main():
    """Main debug function"""
    print("🔬 SpaceGhost: XNNPack Partitioning Debug")
    print("=" * 45)

    # Debug config loading
    our_config = debug_config_loading()

    # Debug target matching
    maxpool_nodes = debug_target_matching()

    # Summary
    print("\n" + "=" * 45)
    print("📊 DEBUG SUMMARY")
    print("=" * 45)

    if our_config and maxpool_nodes:
        print("✅ Config loaded and MaxPool nodes found")
        print("🔧 Issue may be in constraint checking or partitioning logic")
        print("📋 Next: Add debug prints to check_constraints method")
    elif not our_config:
        print("❌ MaxPool2dWithIndicesConfig not properly loaded")
        print("🔧 Fix: Check import statements in __init__.py")
    elif not maxpool_nodes:
        print("❌ No MaxPool nodes found in graph")
        print("🔧 Check: Model may be getting optimized away")
    else:
        print("❓ Unknown issue - need further investigation")

if __name__ == "__main__":
    main()