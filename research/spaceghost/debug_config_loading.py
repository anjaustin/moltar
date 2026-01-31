#!/usr/bin/env python3
"""
Debug if MaxPool2dConfig is loaded in the partitioner.
"""

import sys
import os

# Add ExecuTorch to path
script_dir = os.path.dirname(os.path.abspath(__file__))
executorch_lib = os.path.join(script_dir, 'executorch', 'pip-out', 'lib.macosx-26.0-arm64-cpython-311')
sys.path.insert(0, executorch_lib)

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

def debug_config_loading():
    """Debug if MaxPool2dConfig is loaded"""
    print("🔧 DEBUGGING CONFIG LOADING")
    print("=" * 40)

    # First check what configs are available
    from executorch.backends.xnnpack.partition.config import ALL_PARTITIONER_CONFIGS
    print(f"ALL_PARTITIONER_CONFIGS has {len(ALL_PARTITIONER_CONFIGS)} configs")
    maxpool_configs = [c for c in ALL_PARTITIONER_CONFIGS if 'MaxPool' in c.__name__]
    print(f"MaxPool configs in ALL_PARTITIONER_CONFIGS: {len(maxpool_configs)}")
    for config in maxpool_configs:
        print(f"  {config.__name__}: target_name = {config.target_name}")

        # Fix the target_name if it's wrong
        if config.target_name == "max_pool2d.default":
            print(f"    🔧 Fixing target_name from {config.target_name} to max_pool2d_with_indices.default")
            config.target_name = "max_pool2d_with_indices.default"

    print("\n" + "=" * 40)

    partitioner = XnnpackPartitioner()

    print("Available configs in partitioner:")
    if hasattr(partitioner, 'target_partitioner_configs'):
        for target, config in partitioner.target_partitioner_configs.items():
            print(f"  '{target}' -> {config.__class__.__name__}")

        # Check specifically for MaxPool
        maxpool_configs = {k: v for k, v in partitioner.target_partitioner_configs.items()
                          if 'max_pool' in k}
        print(f"\nMaxPool configs: {len(maxpool_configs)}")
        for target, config in maxpool_configs.items():
            print(f"  '{target}' -> {config.__class__.__name__}")

        expected_targets = ['max_pool2d.default', 'max_pool2d_with_indices.default']
        for target in expected_targets:
            if target in partitioner.target_partitioner_configs:
                config = partitioner.target_partitioner_configs[target]
                print(f"\n{target} config details:")
                print(f"  Class: {config.__class__.__name__}")
                print(f"  target_name: {config.target_name}")
                print(f"  enabled_precision_types: {config.enabled_precision_types}")
            else:
                print(f"\n❌ {target} not found in partitioner!")
    else:
        print("❌ Partitioner has no target_partitioner_configs attribute")

if __name__ == "__main__":
    debug_config_loading()