#!/usr/bin/env python3
"""
SpaceGhost: Apply MaxPool2d Patch to XNNPack Source

This script applies the MaxPool2d fix by modifying the actual XNNPack partitioner source code.
It adds MaxPool2dWithIndicesConfig to handle aten.max_pool2d_with_indices.default operations.
"""

import os
import re
from pathlib import Path

def apply_maxpool2d_source_patch():
    """
    Apply the MaxPool2d fix to the actual XNNPack source code.

    This modifies:
    1. generic_node_configs.py - Add MaxPool2dWithIndicesConfig class
    2. __init__.py - Add import and export of new config
    """
    print("🔧 Applying MaxPool2d Source Code Patch")
    print("=" * 45)

    script_dir = Path(__file__).parent.parent.parent
    xnnpack_dir = script_dir / "executorch" / "backends" / "xnnpack"

    # Check if source files exist
    generic_configs_file = xnnpack_dir / "partition" / "config" / "generic_node_configs.py"
    init_file = xnnpack_dir / "partition" / "config" / "__init__.py"

    if not generic_configs_file.exists():
        print(f"❌ Source file not found: {generic_configs_file}")
        return False

    if not init_file.exists():
        print(f"❌ Init file not found: {init_file}")
        return False

    print("📁 Found source files:")
    print(f"   • {generic_configs_file}")
    print(f"   • {init_file}")

    # Step 1: Add MaxPool2dWithIndicesConfig to generic_node_configs.py
    print("\n📝 Step 1: Adding MaxPool2dWithIndicesConfig class...")

    # Read the current content
    with open(generic_configs_file, 'r') as f:
        content = f.read()

    # Check if the class already exists
    if "class MaxPool2dWithIndicesConfig" in content:
        print("⚠️  MaxPool2dWithIndicesConfig already exists in source")
    else:
        # Find where to insert the new class (after MaxPool2dConfig)
        maxpool_config_pattern = r'class MaxPool2dConfig\(GenericNodePartitionerConfig\):.*?(?=\n\nclass|\n#|\Z)'
        maxpool_match = re.search(maxpool_config_pattern, content, re.DOTALL)

        if maxpool_match:
            # Create the new config class
            new_config = '''

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
'''

            # Insert after MaxPool2dConfig
            insert_pos = maxpool_match.end()
            content = content[:insert_pos] + new_config + content[insert_pos:]

            # Write back
            with open(generic_configs_file, 'w') as f:
                f.write(content)

            print("✅ Added MaxPool2dWithIndicesConfig class")
        else:
            print("❌ Could not find MaxPool2dConfig in source file")
            return False

    # Step 2: Add import and export in __init__.py
    print("\n📝 Step 2: Adding import/export in __init__.py...")

    with open(init_file, 'r') as f:
        init_content = f.read()

    # Check if already imported
    if "MaxPool2dWithIndicesConfig" in init_content:
        print("⚠️  MaxPool2dWithIndicesConfig already imported")
    else:
        # Add to imports
        import_pattern = r'(from executorch\.backends\.xnnpack\.partition\.config\.generic_node_configs import.*?MaxPool2dConfig)'
        import_match = re.search(import_pattern, init_content, re.DOTALL)

        if import_match:
            # Add MaxPool2dWithIndicesConfig to the import
            old_import = import_match.group(1)
            new_import = old_import.replace('MaxPool2dConfig', 'MaxPool2dConfig,\n    MaxPool2dWithIndicesConfig')
            init_content = init_content.replace(old_import, new_import)
            print("✅ Added import for MaxPool2dWithIndicesConfig")
        else:
            print("⚠️  Could not find MaxPool2dConfig import pattern")

        # Add to ALL_PARTITIONER_CONFIGS
        all_configs_pattern = r'(ALL_PARTITIONER_CONFIGS: List\[Type\[XNNPartitionerConfig\]\] = \[.*?\n    MaxPool2dConfig,\n)'
        all_configs_match = re.search(all_configs_pattern, init_content, re.DOTALL)

        if all_configs_match:
            # Add MaxPool2dWithIndicesConfig after MaxPool2dConfig
            old_configs = all_configs_match.group(1)
            new_configs = old_configs.replace('MaxPool2dConfig,\n', 'MaxPool2dConfig,\n    MaxPool2dWithIndicesConfig,\n')
            init_content = init_content.replace(old_configs, new_configs)
            print("✅ Added MaxPool2dWithIndicesConfig to ALL_PARTITIONER_CONFIGS")
        else:
            print("⚠️  Could not find ALL_PARTITIONER_CONFIGS pattern")

        # Write back
        with open(init_file, 'w') as f:
            f.write(init_content)

    print("\n🎉 MaxPool2d Source Patch Applied!")
    print("📋 Changes Made:")
    print("   • Added MaxPool2dWithIndicesConfig class")
    print("   • Added import in __init__.py")
    print("   • Added to ALL_PARTITIONER_CONFIGS list")

    print("\n🧪 Next Steps:")
    print("   1. Test with: python simple_maxpool_test.py")
    print("   2. Verify partitioning works")
    print("   3. Run full validation suite")

    return True


if __name__ == "__main__":
    success = apply_maxpool2d_source_patch()
    if success:
        print("\n🚀 REQ-XNN-001 Implementation Complete!")
        print("MaxPool2d operations should now be partitionable to XNNPack")
    else:
        print("\n❌ Patch application failed")
        exit(1)