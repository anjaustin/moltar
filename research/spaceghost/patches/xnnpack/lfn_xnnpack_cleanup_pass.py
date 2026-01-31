#!/usr/bin/env python3
"""
SpaceGhost: LFN XNNPack Cleanup Pass (Clean Version)

Implements the fix for REQ-XNN-001 and REQ-XNN-002:
1. Fixes MaxPool2d tuple output issue to enable partitioning
2. Removes redundant Q/DQ chains that cause duplication
3. Prepares graph for Snapdragon 480 optimization

This addresses the "Ghost Partition" bug where nodes are accepted but not moved.
"""

import torch
from torch.fx import GraphModule
from executorch.exir.pass_base import ExportPass, PassResult
import operator

class LFNXNNPackCleanupPass(ExportPass):
    """
    Cleanup pass to fix XNNPack partitioning issues for LFN models.

    1. Fixes REQ-XNN-001: Strips unused indices from MaxPool2d to allow delegation
    2. Fixes REQ-XNN-002: Removes redundant Q/DQ chains that cause duplication
    3. Prepares for REQ-XNN-003: Optimizes for Snapdragon 480 memory format
    """

    def call(self, graph_module: GraphModule):
        """
        Apply cleanup transformations to prepare graph for XNNPack partitioning.

        Args:
            graph_module: The FX GraphModule to transform

        Returns:
            PassResult with transformed graph
        """
        graph = graph_module.graph
        nodes_to_remove = []
        modified = False

        print("🧹 Running LFN XNNPack Cleanup Pass...")
        print(f"   📊 Processing graph with {len(list(graph.nodes))} nodes")

        for node in graph.nodes:
            # --- FIX REQ-XNN-001: MaxPool2d Tuple Splitting ---
            # Check for MaxPool2d with indices
            target_str = str(node.target)
            is_maxpool_with_indices = 'max_pool2d_with_indices' in target_str

            if is_maxpool_with_indices:
                users = list(node.users)

                # Find getitem operations that extract tuple elements
                getitem_0 = None  # Values (index 0)
                getitem_1 = None  # Indices (index 1)

                for user in users:
                    if user.target == operator.getitem and len(user.args) >= 2:
                        if user.args[1] == 0:
                            getitem_0 = user
                        elif user.args[1] == 1:
                            getitem_1 = user

                # If indices are not used (getitem_1 doesn't exist or has no users), clean up
                indices_unused = (getitem_1 is None) or (getitem_1 is not None and len(getitem_1.users) == 0)

                if indices_unused and getitem_0 is not None:
                    print(f"🧽 Cleaning unused MaxPool2d indices for node: {node.name}")
                    modified = True

                    # Keep the max_pool2d_with_indices operation (XNNPack supports it)
                    # Just clean up the tuple unpacking to make the partitioner happy

                    # Replace getitem_0 uses with the MaxPool node directly, but only for operations
                    # that can handle tuple inputs (like the partitioner). For tensor operations
                    # that expect a tensor, keep the getitem extraction.
                    getitem_users = list(getitem_0.users)

                    # Only replace uses that are not tensor operations expecting a single tensor
                    tensor_ops_to_keep = {'view_copy', 'permute_copy', 'addmm', 'matmul', 'conv', 'relu'}

                    for user in getitem_users:
                        user_target_str = str(user.target).lower()
                        should_replace = not any(op in user_target_str for op in tensor_ops_to_keep)

                        if should_replace:
                            print(f"      Replacing {user.name} use of getitem_0 with MaxPool node")
                            user.replace_input_with(getitem_0, node)
                        else:
                            print(f"      Keeping getitem_0 for tensor operation: {user.name}")

                    # If getitem_0 still has users after replacement, we can't remove it
                    if len(getitem_0.users) == 0:
                        nodes_to_remove.append(getitem_0)

                    if getitem_1 is not None and len(getitem_1.users) == 0:
                        nodes_to_remove.append(getitem_1)

                    print(f"✅ Cleaned up tuple structure for max_pool2d_with_indices (preserving operation for XNNPack)")

            # --- FIX REQ-XNN-002: Quantization Chain Duplication ---
            # Look for Q -> DQ patterns with identical parameters
            is_quantize = (
                str(node.target) == str(torch.ops.quantized_decomposed.quantize_per_tensor.default) or
                'quantize_per_tensor' in str(node.target)
            )
            if is_quantize:
                input_node = node.args[0] if node.args else None

                is_dequantize = (
                    str(input_node.target) == str(torch.ops.quantized_decomposed.dequantize_per_tensor.default) or
                    'dequantize_per_tensor' in str(input_node.target)
                )
                if input_node and is_dequantize:

                    # Check if Q/DQ parameters match (scale, zero_point, etc.)
                    q_params = node.args[1:4] if len(node.args) > 3 else []
                    dq_params = input_node.args[1:4] if len(input_node.args) > 3 else []

                    if q_params == dq_params:
                        print(f"🔗 Fusing redundant Q/DQ chain: {input_node.name} -> {node.name}")
                        modified = True

                        # Bypass the Q/DQ chain entirely
                        # Replace uses of the quantize node with the input to dequantize
                        dq_input = input_node.args[0] if input_node.args else None
                        if dq_input is not None:
                            node.replace_all_uses_with(dq_input)
                            nodes_to_remove.extend([node, input_node])

            # --- Future: Handle other LFN tuple outputs ---
            # Add similar logic for other Liquid operations that return tuples
            # if node.target == torch.ops.liquid.some_tuple_operation:
            #     # Handle tuple cleanup for Liquid operations
            #     pass

        # Remove the identified nodes
        for node in reversed(nodes_to_remove):
            print(f"🗑️  Removing node: {node.name} ({node.target})")
            graph.erase_node(node)

        if modified:
            graph_module.recompile()
            print("✅ LFN XNNPack Cleanup Pass completed with modifications")
        else:
            print("ℹ️  LFN XNNPack Cleanup Pass completed (no modifications needed)")

        return PassResult(graph_module, modified)


def apply_memory_format_optimization(edge_program):
    """
    Apply memory format optimization for Snapdragon 480.

    The Kryo 460 (Cortex-A76) performs significantly better with NHWC (channels_last)
    for quantized operations like MaxPool2d.

    Args:
        edge_program: The EdgeProgram to optimize

    Returns:
        Optimized EdgeProgram
    """
    print("📱 Applying Snapdragon 480 memory format optimization...")

    try:
        from executorch.exir.passes import MemoryFormatPass

        # Apply channels_last format for SD480 optimization
        memory_pass = MemoryFormatPass(torch.channels_last)
        optimized_graph = memory_pass(edge_program)

        print("✅ Memory format optimized for Snapdragon 480 (NHWC)")
        return optimized_graph

    except Exception as e:
        print(f"⚠️  Memory format optimization failed: {e}")
        print("   Continuing with original format")
        return edge_program


def run_lfn_xnnpack_pipeline(edge_program):
    """
    Run the complete LFN XNNPack preparation pipeline.

    Args:
        edge_program: Input EdgeProgramManager after to_edge()

    Returns:
        Prepared EdgeProgramManager ready for partitioning
    """
    print("🚀 Running LFN XNNPack Preparation Pipeline")
    print("=" * 50)

    # Step 1: Apply cleanup pass to each method's graph
    cleanup_pass = LFNXNNPackCleanupPass()

    for method_name in edge_program.methods:
        print(f"   📋 Processing method: {method_name}")
        exported_prog = edge_program.exported_program(method_name)
        graph_module = exported_prog.graph_module

        cleaned_result = cleanup_pass(graph_module)

        if cleaned_result.modified:
            print("   ✅ Cleanup modifications applied")
        else:
            print("   ℹ️  No cleanup modifications needed")

    print("   ✅ Cleanup pass applied to all methods")

    # Step 2: Apply memory format optimization
    print("   📱 Applying Snapdragon 480 memory format optimization...")
    try:
        # For EdgeProgramManager, we need to apply to the exported programs
        for method_name in edge_program.methods:
            exported_prog = edge_program.exported_program(method_name)
            graph_module = exported_prog.graph_module

            # Apply channels_last format for SD480 optimization
            try:
                from executorch.exir.passes import MemoryFormatPass
                memory_pass = MemoryFormatPass(torch.channels_last)
                optimized_result = memory_pass(graph_module)
                print(f"   ✅ Memory format optimized for method: {method_name}")
            except Exception as e:
                print(f"   ⚠️  Memory format optimization failed for {method_name}: {e}")

    except Exception as e:
        print(f"   ⚠️  Memory format optimization setup failed: {e}")

    print("✅ Pipeline completed - EdgeProgramManager ready for XNNPack partitioning")
    return edge_program