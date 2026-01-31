#!/usr/bin/env python3
"""
SpaceGhost: Manual Delegation - Nuclear Option

Since the partitioner creates tags but fails to delegate, manually create
the delegate nodes by manipulating the graph directly.
"""

import sys
import os

# Add ExecuTorch to path
script_dir = os.path.dirname(os.path.abspath(__file__))
executorch_lib = os.path.join(script_dir, 'executorch', 'pip-out', 'lib.macosx-26.0-arm64-cpython-311')
sys.path.insert(0, executorch_lib)

import torch
from executorch.exir.backend.backend_api import to_backend
from executorch.backends.xnnpack.xnnpack_preprocess import XnnpackBackend
from executorch.exir.backend.backend_details import CompileSpec

def manually_create_delegates(partitioned_result):
    """
    Manually create delegate nodes for tagged operations that weren't moved.

    This is the ultimate workaround for the "Ghost Partition" bug.
    """
    print("🔧 MANUAL DELEGATION: Creating delegate nodes for tagged operations")
    print("=" * 70)

    if not hasattr(partitioned_result, 'tagged_exported_program'):
        print("❌ No tagged_exported_program found")
        return partitioned_result

    tagged_ep = partitioned_result.tagged_exported_program
    graph = tagged_ep.graph_module.graph

    # Get partition tags
    partition_tags = getattr(partitioned_result, 'partition_tags', {})
    print(f"📋 Found {len(partition_tags)} partition tags")

    for tag_name, partition_spec in partition_tags.items():
        print(f"\n🎯 Processing partition: {tag_name}")
        print(f"   Backend: {partition_spec.backend_id}")

        # Find nodes with this partition tag
        tagged_nodes = []
        for node in graph.nodes:
            if hasattr(node, 'meta') and 'partition_tag' in node.meta:
                if node.meta['partition_tag'] == tag_name:
                    tagged_nodes.append(node)

        print(f"   Tagged nodes: {len(tagged_nodes)}")
        for node in tagged_nodes:
            print(f"      • {node.name}: {node.target}")

        if not tagged_nodes:
            print("   ⚠️  No tagged nodes found, skipping")
            continue

        # Create a lowered module for these nodes
        try:
            # Use the backend API to lower the tagged nodes
            compile_specs = [CompileSpec("target", "arm64-v8.2-a+dotprod")]
            lowered_module = to_backend(
                partition_spec.backend_id,
                tagged_ep,  # Use the tagged exported program
                tagged_nodes,  # The nodes to lower
                compile_specs
            )

            print("   ✅ Lowered module created")

            # Replace the tagged nodes with a call to the lowered module
            first_node = tagged_nodes[0]

            with graph.inserting_before(first_node):
                # Create delegate call
                delegate_call = graph.call_function(
                    lowered_module,
                    args=first_node.args,
                    kwargs=first_node.kwargs
                )
                delegate_call.name = f"executorch_call_delegate_{tag_name}"

                # Redirect all users of the last tagged node to the delegate
                last_node = tagged_nodes[-1]
                last_node.replace_all_uses_with(delegate_call)

            # Remove the original tagged nodes
            for node in reversed(tagged_nodes):
                print(f"   🗑️  Removing tagged node: {node.name}")
                graph.erase_node(node)

            print(f"   🎉 Successfully created delegate for {tag_name}")

        except Exception as e:
            print(f"   ❌ Failed to create delegate for {tag_name}: {e}")
            import traceback
            traceback.print_exc()

    # Recompile the graph
    tagged_ep.graph_module.recompile()
    print(f"\n✅ Manual delegation completed")

    return partitioned_result

def test_manual_delegation():
    """Test manual delegation after partitioning"""
    print("🧪 TESTING MANUAL DELEGATION")
    print("=" * 50)

    from torch.export import export
    from executorch.exir import to_edge
    from patches.xnnpack.lfn_xnnpack_cleanup_pass import run_lfn_xnnpack_pipeline

    # Create test model
    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1)
            self.pool1 = torch.nn.MaxPool2d(2, 2)
            self.conv2 = torch.nn.Conv2d(64, 128, 3, 1, 1)
            self.pool2 = torch.nn.MaxPool2d(2, 2)
            self.fc = torch.nn.Linear(128 * 8 * 8, 10)

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

    print("3. Applying XNNPack partitioner...")
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

    partitioner = XnnpackPartitioner()
    method_name = list(cleaned_edge.methods)[0]
    exported_prog = cleaned_edge.exported_program(method_name)

    partitioned = partitioner(exported_prog)

    print("4. Checking partitioner results...")
    if hasattr(partitioned, 'tagged_exported_program'):
        tagged_ep = partitioned.tagged_exported_program
        pre_manual_ops = [node for node in tagged_ep.graph_module.graph.nodes if node.op == 'call_function']
        pre_manual_maxpool = [op for op in pre_manual_ops if 'max_pool' in str(op.target)]
        pre_manual_delegates = [op for op in pre_manual_ops if 'delegate' in str(op.target).lower()]

        print(f"   📊 After partitioner: {len(pre_manual_ops)} ops, {len(pre_manual_maxpool)} MaxPool, {len(pre_manual_delegates)} delegates")

        print("\n5. APPLYING MANUAL DELEGATION...")
        manually_delegated = manually_create_delegates(partitioned)

        print("\n6. Checking final results...")
        final_ep = manually_delegated.tagged_exported_program
        final_ops = [node for node in final_ep.graph_module.graph.nodes if node.op == 'call_function']
        final_maxpool = [op for op in final_ops if 'max_pool' in str(op.target)]
        final_delegates = [op for op in final_ops if 'delegate' in str(op.target).lower()]

        print(f"   📊 After manual delegation: {len(final_ops)} ops, {len(final_maxpool)} MaxPool, {len(final_delegates)} delegates")

        if final_delegates:
            print("   🎯 Delegate nodes created:")
            for delegate in final_delegates:
                print(f"      • {delegate.name}: {delegate.target}")

        print("\n" + "=" * 70)
        print("🎯 MANUAL DELEGATION VERDICT")
        print("=" * 70)

        if len(final_maxpool) < len(pre_manual_maxpool) and len(final_delegates) > 0:
            print("🎉 SUCCESS: Manual delegation worked!")
            print(f"   • MaxPool ops reduced: {len(pre_manual_maxpool)} → {len(final_maxpool)}")
            print(f"   • Delegate nodes created: {len(final_delegates)}")
            print("   • REQ-XNN-001: SOLVED via manual delegation")
            return True
        else:
            print("❌ FAILURE: Manual delegation did not work")
            print(f"   • MaxPool ops: {len(pre_manual_maxpool)} → {len(final_maxpool)}")
            print(f"   • Delegates: {len(pre_manual_delegates)} → {len(final_delegates)}")
            return False
    else:
        print("❌ No tagged_exported_program after partitioning")
        return False

if __name__ == "__main__":
    success = test_manual_delegation()
    sys.exit(0 if success else 1)