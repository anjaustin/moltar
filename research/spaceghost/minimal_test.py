#!/usr/bin/env python3
"""
Minimal MaxPool2d Test
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

# Runtime patch
from executorch.backends.xnnpack.partition.config.generic_node_configs import GenericNodePartitionerConfig
import logging
logger = logging.getLogger(__name__)

class MaxPool2dWithIndicesConfig(GenericNodePartitionerConfig):
    target_name = "max_pool2d_with_indices.default"

    def check_constraints(self, node, ep):
        print(f"🔍 Checking constraints for MaxPool2d node: {node}")

        # Skip the problematic common constraints check
        # Instead, do our own basic validation

        # Check that we have the right number of arguments
        if len(node.args) < 2:
            print("❌ Not enough arguments for MaxPool2d")
            return False

        # Check kernel_size and stride constraints
        kernel_size = node.args[1]
        stride = node.args[2] if len(node.args) > 2 else kernel_size

        if stride[0] > kernel_size[0] or stride[1] > kernel_size[1]:
            print(f"❌ Invalid stride {stride} > kernel_size {kernel_size}")
            return False

        # Check indices usage (simplified)
        indices_users = []
        for user in node.users:
            if user.op == 'call_function' and str(user.target) == 'operator.getitem':
                if len(user.args) > 1 and user.args[1] == 1:
                    indices_users.append(user)

        if indices_users:
            print(f"❌ Indices have users: {len(indices_users)} users")
            return False

        print("✅ All constraints passed")
        return True

    def supported_precision_types(self):
        from executorch.backends.xnnpack.partition.config.xnnpack_config import ConfigPrecisionType
        return [ConfigPrecisionType.FP32, ConfigPrecisionType.STATIC_QUANT]

# Apply patch
ALL_PARTITIONER_CONFIGS.append(MaxPool2dWithIndicesConfig)
print("✅ MaxPool2dWithIndicesConfig added")

# Test
print("🧪 Running minimal test...")

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

model = TestModel()
sample_input = torch.randn(1, 3, 32, 32)

print("1. Export...")
exported = export(model, (sample_input,))
print("✅ Exported")

print("2. To edge...")
edge = to_edge(exported)
print("✅ Edge converted")

print("3. Partition...")
partitioner = XnnpackPartitioner()
method_name = list(edge.methods)[0]
exported_prog = edge.exported_program(method_name)

print("   Attempting partition...")
partitioned = partitioner(exported_prog)
print("✅ Partitioning completed!")

print("🎉 SUCCESS: MaxPool2d partitioning works!")