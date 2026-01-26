#!/usr/bin/env python3
"""
Test with forced module reload
"""

import sys
import importlib

# Add ExecuTorch to path
script_dir = '/Users/aaronjosserand-austin/000/Motorola/research/spaceghost'
executorch_lib = f'{script_dir}/executorch/pip-out/lib.macosx-26.0-arm64-cpython-311'
sys.path.insert(0, executorch_lib)

# Force reload of modules
modules_to_reload = [
    'executorch.backends.xnnpack.partition.config.generic_node_configs',
    'executorch.backends.xnnpack.partition.config',
    'executorch.backends.xnnpack.partition.xnnpack_partitioner',
]

for module_name in modules_to_reload:
    if module_name in sys.modules:
        print(f"Reloading {module_name}")
        importlib.reload(sys.modules[module_name])

# Now test
from executorch.backends.xnnpack.partition.config import ALL_PARTITIONER_CONFIGS

print(f"Total configs: {len(ALL_PARTITIONER_CONFIGS)}")

maxpool_configs = [c for c in ALL_PARTITIONER_CONFIGS if 'MaxPool' in c.__name__]
print(f"MaxPool configs: {[c.__name__ for c in maxpool_configs]}")

our_config = None
for config in ALL_PARTITIONER_CONFIGS:
    if config.__name__ == 'MaxPool2dWithIndicesConfig':
        our_config = config
        break

if our_config:
    print("✅ SUCCESS: MaxPool2dWithIndicesConfig found!")
    print(f"   Target: {our_config.target_name}")
else:
    print("❌ FAILURE: MaxPool2dWithIndicesConfig still not found")

# Test partitioning
import torch
import torch.nn as nn
from torch.export import export

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

from executorch.exir import to_edge
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

exported = export(model, (sample_input,))
edge = to_edge(exported)

partitioner = XnnpackPartitioner()
method_name = list(edge.methods)[0]
exported_prog = edge.exported_program(method_name)

print("Running partitioning...")
partitioned = partitioner(exported_prog)

# Check results
if hasattr(partitioned, 'tagged_exported_program'):
    tagged_ep = partitioned.tagged_exported_program
    ops = []
    for node in tagged_ep.graph_module.graph.nodes:
        if node.op == 'call_function':
            ops.append(str(node.target))

    maxpool_ops = [op for op in ops if 'max_pool' in op.lower()]
    if maxpool_ops:
        print(f"❌ MaxPool still in main graph: {maxpool_ops}")
    else:
        print("✅ SUCCESS: MaxPool2d partitioned to XNNPack!")
else:
    print("❌ Partitioning failed")