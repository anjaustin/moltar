#!/usr/bin/env python3
"""
SpaceGhost: Debug Config Matching

Investigate why MaxPool2dWithIndicesConfig is not being matched during partitioning.
Check the config loading, target name matching, and constraint evaluation.
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
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.partition.config import ALL_PARTITIONER_CONFIGS

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 16 * 16, 10)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        return self.fc(x.view(x.size(0), -1))

def debug_config_loading():
    """Debug what configs are loaded"""
    print("🔍 Checking loaded configs...")

    # Find our config
    our_config = None
    for config in ALL_PARTITIONER_CONFIGS:
        if hasattr(config, '__name__') and 'MaxPool2dWithIndices' in config.__name__:
            our_config = config
            break

    if our_config:
        print(f"✅ Found our config: {our_config}")
        print(f"   target_name: {our_config.target_name}")
        config_instance = our_config()
        print(f"   supported precisions: {config_instance.supported_precision_types()}")
    else:
        print("❌ MaxPool2dWithIndicesConfig not found!")
        print("Available configs:")
        for config in ALL_PARTITIONER_CONFIGS:
            if hasattr(config, '__name__') and 'MaxPool' in config.__name__:
                print(f"   • {config.__name__}")

def debug_target_matching():
    """Debug target name matching process"""
    print("\n🔍 Testing target matching...")

    model = TestModel()
    sample_input = torch.randn(1, 3, 32, 32)

    exported = export(model, (sample_input,))
    edge = to_edge(exported)

    method_name = list(edge.methods)[0]
    exported_prog = edge.exported_program(method_name)
    graph_module = exported_prog.graph_module

    print("MaxPool nodes in graph:")
    for node in graph_module.graph.nodes:
        if node.op == 'call_function' and 'max_pool' in str(node.target).lower():
            print(f"   📍 {node}")
            print(f"      target: {node.target}")
            print(f"      target.__name__: {node.target.__name__}")
            print(f"      args: {node.args}")

            # Test format_target_name
            from executorch.exir.backend.canonical_partitioners.config_partitioner import format_target_name
            formatted = format_target_name(node.target.__name__)
            print(f"      formatted target: '{formatted}'")

            # Check which config would match
            matching_configs = []
            for config in ALL_PARTITIONER_CONFIGS:
                if hasattr(config, 'target_name') and config.target_name == formatted:
                    matching_configs.append(config)

            if matching_configs:
                print(f"      ✅ MATCHES: {[c.__name__ for c in matching_configs]}")
            else:
                print(f"      ❌ NO MATCH for '{formatted}'")

                # Show similar configs
                similar = [c for c in ALL_PARTITIONER_CONFIGS
                          if hasattr(c, 'target_name') and 'max_pool' in c.target_name.lower()]
                if similar:
                    print(f"      💡 Similar configs: {[c.__name__ for c in similar]}")

def debug_partitioning_process():
    """Debug the actual partitioning process step by step"""
    print("\n🔍 Debugging partitioning process...")

    model = TestModel()
    sample_input = torch.randn(1, 3, 32, 32)

    exported = export(model, (sample_input,))
    edge = to_edge(exported)

    method_name = list(edge.methods)[0]
    exported_prog = edge.exported_program(method_name)

    print("Creating partitioner...")
    partitioner = XnnpackPartitioner()

    # Check partitioner config
    print(f"Partitioner has {len(partitioner._target_partitioner_configs)} target configs")

    # Try to manually check what would be partitioned
    print("Manually checking partitionable nodes...")
    for node in exported_prog.graph_module.graph.nodes:
        if node.op == 'call_function':
            target_name = node.target.__name__
            from executorch.exir.backend.canonical_partitioners.config_partitioner import format_target_name
            formatted = format_target_name(target_name)

            if formatted in partitioner._target_partitioner_configs:
                config = partitioner._target_partitioner_configs[formatted]
                print(f"   🎯 {target_name} -> {formatted} matches {config.__class__.__name__}")

                # Test constraints
                try:
                    can_partition = config.check_constraints(node, exported_prog)
                    print(f"      Constraint check: {'✅ PASS' if can_partition else '❌ FAIL'}")
                except Exception as e:
                    print(f"      Constraint check ERROR: {e}")
            else:
                print(f"   ➖ {target_name} -> {formatted} no match")

    print("Running actual partitioning...")
    try:
        partitioned = partitioner(exported_prog)
        print("✅ Partitioning completed")

        if hasattr(partitioned, 'tagged_exported_program'):
            print("Checking results...")
            tagged_ep = partitioned.tagged_exported_program
            maxpool_count = 0
            for node in tagged_ep.graph_module.graph.nodes:
                if node.op == 'call_function' and 'max_pool' in str(node.target).lower():
                    maxpool_count += 1
                    print(f"   ⚠️  MaxPool still in main graph: {node.target}")

            if maxpool_count == 0:
                print("   ✅ No MaxPool nodes in main graph - successfully partitioned!")
            else:
                print(f"   ❌ {maxpool_count} MaxPool nodes still in main graph")

    except Exception as e:
        print(f"❌ Partitioning failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main debug function"""
    print("🔬 SpaceGhost: Config Matching Debug")
    print("=" * 45)

    debug_config_loading()
    debug_target_matching()
    debug_partitioning_process()

    print("\n" + "=" * 45)
    print("🎯 DEBUG SUMMARY")
    print("=" * 45)
    print("If MaxPool2dWithIndicesConfig exists but isn't matching:")
    print("1. Check target_name formatting")
    print("2. Verify config is in _target_partitioner_configs")
    print("3. Test constraint checking manually")
    print("4. Check for import/loading issues")

if __name__ == "__main__":
    main()