#!/usr/bin/env python3
"""
Debug the ExecuTorch partitioner to understand why it fails to tag nodes.
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
from executorch.exir.backend.canonical_partitioners.config_partitioner import format_target_name

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.pool(x)

def debug_partitioner_matching():
    """Debug why the partitioner fails to match our nodes"""

    print("🔧 DEBUGGING PARTITIONER NODE MATCHING")
    print("=" * 50)

    # Create and process model
    model = TestModel()
    sample_input = torch.randn(1, 3, 32, 32)

    exported = export(model, (sample_input,))
    edge = to_edge(exported)
    cleaned_edge = run_lfn_xnnpack_pipeline(edge)

    method_name = list(cleaned_edge.methods)[0]
    exported_prog = cleaned_edge.exported_program(method_name)

    print("📊 NODES IN PROCESSED GRAPH:")
    for node in exported_prog.graph_module.graph.nodes:
        if node.op == 'call_function':
            target_name = format_target_name(node.target.__name__)
            print(f"  {node.name}: {node.target} -> formatted: '{target_name}'")

    print("\n🎯 PARTITIONER CONFIG TARGETS:")
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    partitioner = XnnpackPartitioner()

    # Access the internal config map
    if hasattr(partitioner, 'target_partitioner_configs'):
        for target, config in partitioner.target_partitioner_configs.items():
            print(f"  '{target}' -> {config.__class__.__name__}")
    else:
        print("  ❌ Cannot access partitioner configs")

    print("\n🔍 MANUAL NODE MATCHING TEST:")
    from executorch.backends.xnnpack.partition.config.generic_node_configs import MaxPool2dConfig
    config = MaxPool2dConfig()
    print(f"Config target_name: '{config.target_name}'")

    matched_nodes = []
    for node in exported_prog.graph_module.graph.nodes:
        if node.op == 'call_function':
            target_name = format_target_name(node.target.__name__)
            matches = target_name == config.target_name
            print(f"  {node.name}: '{target_name}' == '{config.target_name}' -> {matches}")
            if matches:
                constraint_check = config.check_constraints(node, exported_prog)
                print(f"    Constraint check: {constraint_check}")
                if constraint_check:
                    matched_nodes.append(node)

    print(f"\n🎉 MANUAL MATCHING FOUND: {len(matched_nodes)} nodes")
    for node in matched_nodes:
        print(f"  • {node.name}: {node.target}")

    print("\n🚀 TESTING PARTITIONER DIRECTLY:")
    result = partitioner(exported_prog)
    print(f"Partition tags: {len(result.partition_tags) if result.partition_tags else 0}")
    if result.partition_tags:
        for tag, spec in result.partition_tags.items():
            print(f"  {tag}: {spec.backend_id}")

    # Check if any nodes were actually tagged
    tagged_nodes = []
    for node in exported_prog.graph_module.graph.nodes:
        if hasattr(node, 'meta') and 'delegation_tag' in node.meta:
            tagged_nodes.append(node)

    print(f"Actually tagged nodes: {len(tagged_nodes)}")
    for node in tagged_nodes:
        print(f"  • {node.name}: tag='{node.meta['delegation_tag']}'")

if __name__ == "__main__":
    debug_partitioner_matching()