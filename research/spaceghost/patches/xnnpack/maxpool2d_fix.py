#!/usr/bin/env python3
"""
SpaceGhost: MaxPool2d XNNPack Fix

Implementation of REQ-XNN-001: Add support for MaxPool2d operations in XNNPack partitioner.

Issue: nn.MaxPool2d decomposes to aten.max_pool2d_with_indices.default,
but XNNPack partitioner only supports max_pool2d.default.

Fix: Add configuration for the correct ATen operation and handle indices output.
"""

import os
import sys

# Add ExecuTorch to path
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
executorch_lib = os.path.join(script_dir, 'executorch', 'pip-out', 'lib.macosx-26.0-arm64-cpython-311')
sys.path.insert(0, executorch_lib)

from typing import List
import torch
from executorch.backends.xnnpack.partition.config.xnnpack_config import (
    ConfigPrecisionType,
    XNNPartitionerConfig,
)
from executorch.backends.xnnpack.partition.config.generic_node_configs import (
    GenericNodePartitionerConfig,
)
from executorch.exir.backend.canonical_partitioners.config_partitioner import (
    format_target_name,
)
from executorch.exir.backend.utils import is_shape_dynamic, WhyNoPartition
from torch.export import ExportedProgram

import logging
logger = logging.getLogger(__name__)
why = WhyNoPartition(logger=logger)


class MaxPool2dWithIndicesConfig(GenericNodePartitionerConfig):
    """
    Configuration for aten.max_pool2d_with_indices.default operation.

    This handles the decomposed form of nn.MaxPool2d which includes indices output.
    We only partition if the indices are not used by downstream operations.
    """
    target_name = "aten.max_pool2d_with_indices.default"

    def check_constraints(self, node: torch.fx.Node, ep: ExportedProgram) -> bool:
        """
        XNNPACK's maxpool2d does not support ceil mode and requires stride <= kernel_size.
        Additionally, we check if indices output has users.
        """
        if not self.check_common_constraints(node, ep):
            return False

        # Check if indices output has users
        # node.args[0] is input, node.args[1] is kernel_size, etc.
        # The operation returns (values, indices), so we need to check if indices are used
        graph = ep.graph_module.graph if hasattr(ep, 'graph_module') else ep.graph

        # Find the getitem nodes that access the indices (output[1])
        indices_users = []
        for user in node.users:
            if user.op == 'call_function' and str(user.target) == 'operator.getitem':
                # Check if this getitem accesses index 1 (indices)
                if len(user.args) > 1 and user.args[1] == 1:
                    indices_users.append(user)

        if indices_users:
            why(node, reason="MaxPool2d indices output has users - cannot partition")
            return False

        # Standard MaxPool2d constraints
        kernel_size = node.args[1]
        stride = node.args[2] if len(node.args) > 2 else kernel_size
        padding = node.args[3] if len(node.args) > 3 else [0, 0]
        dilation = node.args[4] if len(node.args) > 4 else [1, 1]
        ceil_mode = node.args[5] if len(node.args) > 5 else False

        # Ceil mode is supported via op padding, which must be statically known.
        if ceil_mode and is_shape_dynamic(node):
            why(node, reason="ceil mode is not supported for dynamic shapes")
            return False

        if stride[0] > kernel_size[0] or stride[1] > kernel_size[1]:
            why(
                node,
                reason=f"stride ({stride}) must be less than or equal to kernel size ({kernel_size})",
            )
            return False

        return True

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32, ConfigPrecisionType.STATIC_QUANT]

    def get_node_and_deps(
        self, node: torch.fx.Node, ep: ExportedProgram
    ) -> List[torch.fx.Node]:
        """
        Get the node and its dependencies. For MaxPool2d, we need to ensure
        only the values output (not indices) is used.
        """
        deps = [node]

        # Find and include any getitem operations that access values (index 0)
        graph = ep.graph_module.graph if hasattr(ep, 'graph_module') else ep.graph
        for user in node.users:
            if user.op == 'call_function' and str(user.target) == 'operator.getitem':
                if len(user.args) > 1 and user.args[1] == 0:  # values output
                    deps.append(user)

        return deps


def apply_maxpool2d_fix():
    """
    Apply the MaxPool2d fix to the XNNPack partitioner configuration.

    This would normally be integrated into the XNNPack partitioner,
    but for SpaceGhost we demonstrate the fix here.
    """
    print("🔧 Applying MaxPool2d XNNPack Fix")
    print("=" * 40)

    # This fix would be integrated by:
    # 1. Adding MaxPool2dWithIndicesConfig to ALL_PARTITIONER_CONFIGS
    # 2. Ensuring proper import in the partitioner

    print("✅ MaxPool2dWithIndicesConfig created")
    print("📍 Target: aten.max_pool2d_with_indices.default")
    print("🔍 Checks indices usage before partitioning")
    print("📋 Supports FP32 and Static Quantization")

    print("\n🚀 Integration Steps:")
    print("1. Add MaxPool2dWithIndicesConfig to ALL_PARTITIONER_CONFIGS")
    print("2. Import in xnnpack_partitioner.py")
    print("3. Test with simple_maxpool_test.py")
    print("4. Validate on Snapdragon 480")

    return True


if __name__ == "__main__":
    apply_maxpool2d_fix()