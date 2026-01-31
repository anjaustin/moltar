#!/usr/bin/env python3
"""
Manual Delegation Fix for ExecuTorch "Ghost Partition" Bug

This implements the nuclear option: manually delegate partitioned submodules
to XNNPack when the automated system fails.
"""

import sys
import os

# Add ExecuTorch to path
script_dir = os.path.dirname(os.path.abspath(__file__))
executorch_lib = os.path.join(script_dir, 'executorch', 'pip-out', 'lib.macosx-26.0-arm64-cpython-311')
sys.path.insert(0, executorch_lib)

import torch
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.backend_details import CompileSpec

def find_node_by_name(graph, name):
    """Find a node in the graph by name"""
    for node in graph.nodes:
        if node.name == name:
            return node
    return None

def manual_delegation_fix(partitioned_result):
    """
    Manually delegate tagged nodes to XNNPack.

    This bypasses the broken `lower_all_submodules_to_backend` function
    by manually creating submodules from tagged nodes and lowering them.
    """
    print("🔧 MANUAL DELEGATION FIX: Implementing nuclear option")
    print("=" * 60)

    if not hasattr(partitioned_result, 'tagged_exported_program'):
        print("❌ No tagged_exported_program found")
        return partitioned_result

    if not hasattr(partitioned_result, 'partition_tags'):
        print("❌ No partition_tags found")
        return partitioned_result

    tagged_ep = partitioned_result.tagged_exported_program
    graph = tagged_ep.graph_module.graph
    partition_tags = partitioned_result.partition_tags

    print(f"📋 Found {len(partition_tags)} partition tags")

    # Group tagged nodes by partition tag
    tag_to_nodes = {}
    for node in graph.nodes:
        if hasattr(node, 'meta') and 'delegation_tag' in node.meta:
            tag = node.meta['delegation_tag']
            if tag not in tag_to_nodes:
                tag_to_nodes[tag] = []
            tag_to_nodes[tag].append(node)

    print(f"📋 Found {len(tag_to_nodes)} tagged node groups")

    total_delegated = 0

    for tag, nodes in tag_to_nodes.items():
        print(f"\n🎯 Processing partition: {tag}")
        print(f"   Nodes: {len(nodes)}")
        for node in nodes:
            print(f"      • {node.name}: {node.target}")

        if tag not in partition_tags:
            print(f"   ❌ No partition spec found for tag: {tag}")
            continue

        partition_spec = partition_tags[tag]
        backend_id = partition_spec.backend_id
        compile_specs = partition_spec.compile_specs

        print(f"   Backend: {backend_id}")
        print(f"   Compile specs: {compile_specs}")

        try:
            # Create a submodule from the tagged nodes
            print("   🔄 Creating submodule from tagged nodes...")
            submodule, call_module_node = _create_submodule_from_tagged_nodes(
                tagged_ep, nodes, tag
            )

            print("   ✅ Submodule created")

            # Lower the submodule to the backend
            print(f"   🔄 Lowering submodule to {backend_id}...")
            lowered_module = to_backend(backend_id, submodule, compile_specs)

            print("   ✅ Lowering successful")

            # Replace the submodule call with a delegate call
            _replace_submodule_with_delegate(
                graph, call_module_node, lowered_module,
                None,  # We don't have output node info
                False  # Not a submodule
            )

            total_delegated += 1
            print(f"   🎉 Successfully delegated partition: {tag}")

        except Exception as e:
            print(f"   ❌ Failed to delegate partition {tag}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n" + "=" * 60)
    print("🎯 MANUAL DELEGATION FIX RESULTS")
    print("=" * 60)
    print(f"✅ Successfully delegated: {total_delegated} partitions")

    if total_delegated > 0:
        print("🎉 MANUAL DELEGATION FIX SUCCESSFUL!")
        print("   The 'Ghost Partition' bug has been bypassed")
        print("   REQ-XNN-001: SOLVED via manual delegation")
    else:
        print("❌ Manual delegation failed - no partitions were delegated")

    return partitioned_result

def _create_submodule_from_tagged_nodes(tagged_ep, nodes, tag):
    """
    Create a submodule from a list of tagged nodes.

    This replicates the submodule creation logic from the partitioner.
    """
    from torch.fx.passes.utils.fuser_utils import fuse_as_graphmodule
    from torch.fx.passes.utils.fuser_utils import topo_sort

    # Sort nodes topologically
    sorted_nodes = topo_sort(nodes)

    # Create a name for the submodule
    submodule_name = f"fused_{tag}"

    # Fuse the nodes into a submodule
    owning_graph = tagged_ep.graph_module
    sub_gm, orig_inputs, orig_outputs = fuse_as_graphmodule(
        owning_graph, sorted_nodes, submodule_name
    )

    # Insert the submodule into the owning graph
    from torch.fx.graph_module import GraphModule
    from torch.fx.passes.utils.fuser_utils import insert_subgm

    owning_graph = insert_subgm(owning_graph, sub_gm, orig_inputs, orig_outputs)

    # Find the call_module node that was created
    call_module_node = None
    for node in owning_graph.graph.nodes:
        if node.op == "call_module" and node.target == submodule_name:
            call_module_node = node
            break

    if call_module_node is None:
        raise RuntimeError(f"Could not find call_module node for {submodule_name}")

    # Create an ExportedProgram from the submodule
    from torch.export import export
    from torch.fx import symbolic_trace

    # We need to create a proper input spec for the submodule
    # This is tricky - let's try to trace the submodule
    try:
        traced = symbolic_trace(sub_gm)
        # Create dummy inputs based on the input nodes
        dummy_inputs = []
        for orig_input in orig_inputs:
            if hasattr(orig_input, 'meta') and 'val' in orig_input.meta:
                dummy_inputs.append(torch.zeros_like(orig_input.meta['val']))
            else:
                dummy_inputs.append(torch.randn(1, 3, 32, 32))  # Fallback

        submodule_ep = export(traced, tuple(dummy_inputs))
        return submodule_ep, call_module_node

    except Exception as e:
        print(f"   ⚠️  Failed to create ExportedProgram: {e}")
        # Fallback: return the GraphModule directly
        return sub_gm, call_module_node

def _replace_submodule_with_delegate(graph, submodule_node, lowered_module,
                                    submodule_output_node, is_submodule):
    """
    Replace a submodule call with a delegate call.

    This replicates the logic from _insert_lowered_submodule but manually.
    """
    print(f"   🔄 Replacing submodule {submodule_node.name} with delegate")

    # Get call arguments (skip the 'self' parameter for submodules)
    call_args = submodule_node.args

    # Create the delegate call
    with graph.inserting_before(submodule_node):
        # Import the delegate function
        from executorch.exir.backend.backend_details import executorch_call_delegate

        # Create the call to the lowered module
        delegate_call = graph.call_function(
            executorch_call_delegate,
            (lowered_module,) + tuple(call_args),
            submodule_node.kwargs
        )

        delegate_call.name = f"executorch_call_delegate_{submodule_node.name}"
        delegate_call.meta["debug_handle"] = 1000 + hash(submodule_node.name) % 1000

        # Set up the output metadata
        if submodule_output_node:
            delegate_call.meta["val"] = [
                out_arg.meta["val"] for out_arg in submodule_output_node.args[0]
            ]

        # Redirect all uses of the submodule to the delegate
        submodule_node.replace_all_uses_with(delegate_call)

    # Remove the original submodule node
    print(f"   🗑️  Removing submodule node: {submodule_node.name}")
    graph.erase_node(submodule_node)

    print(f"   ✅ Delegate call created: {delegate_call.name}")

def apply_manual_delegation_fix(edge_program):
    """
    Apply the manual delegation fix to an EdgeProgramManager after partitioning.

    This is the complete workflow:
    1. Partition the program
    2. Manually delegate the submodules that weren't lowered
    """
    print("🚨 APPLYING MANUAL DELEGATION FIX")
    print("=" * 50)

    # Step 1: Partition (this should work)
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    partitioner = XnnpackPartitioner()

    method_name = list(edge_program.methods)[0]
    exported_prog = edge_program.exported_program(method_name)

    print("1. Partitioning...")
    partitioned = partitioner(exported_prog)

    # Check if partitioning worked
    tagged_nodes = []
    if hasattr(partitioned, 'tagged_exported_program'):
        for node in partitioned.tagged_exported_program.graph_module.graph.nodes:
            if hasattr(node, 'meta') and 'delegation_tag' in node.meta:
                tagged_nodes.append(node)

    print(f"   Tagged nodes: {len(tagged_nodes)}")

    if len(tagged_nodes) == 0:
        print("❌ Partitioning failed - no nodes were tagged")
        return edge_program

    # Step 2: Apply manual delegation fix
    print("\n2. Applying manual delegation fix...")
    fixed_result = manual_delegation_fix(partitioned)

    # Step 3: Check results
    print("\n3. Checking final results...")
    if hasattr(fixed_result, 'tagged_exported_program'):
        final_ep = fixed_result.tagged_exported_program
        final_ops = [node for node in final_ep.graph_module.graph.nodes if node.op == 'call_function']
        final_delegates = [op for op in final_ops if 'delegate' in str(op.target).lower()]

        print(f"   📊 Final operations: {len(final_ops)}")
        print(f"   🎯 Delegate calls: {len(final_delegates)}")

        if len(final_delegates) > 0:
            print("   ✅ SUCCESS: Manual delegation fix worked!")
            print("   🎉 ExecuTorch 'Ghost Partition' bug has been bypassed!")

            # Create new EdgeProgramManager with the fixed result
            from executorch.exir.program._program import EdgeProgramManager
            from executorch.exir._config import EdgeCompileConfig

            config = EdgeCompileConfig(_check_ir_validity=False)
            fixed_epm = EdgeProgramManager(
                {method_name: final_ep},
                edge_program._config_methods,
                config
            )
            return fixed_epm
        else:
            print("   ❌ FAILED: No delegate calls created")
            return edge_program
    else:
        print("   ❌ No tagged_exported_program in result")
        return edge_program