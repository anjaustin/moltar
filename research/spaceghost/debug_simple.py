#!/usr/bin/env python3
"""
SpaceGhost: Simple ExecuTorch Diagnostic Script

Basic diagnostic to check if ExecuTorch is available and working.
"""

import sys
import os

def check_executorch_availability():
    """Check if ExecuTorch modules are available"""
    print("🔍 SpaceGhost: Basic ExecuTorch Diagnostic")
    print("=" * 50)

    # Add executorch to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'executorch'))

    modules_to_check = [
        'torch',
        'torch.export',
        'executorch.exir',
        'executorch.backends.xnnpack_partitioner',
        'executorch.sdk'
    ]

    available_modules = []
    missing_modules = []

    for module in modules_to_check:
        try:
            __import__(module)
            available_modules.append(module)
            print(f"✅ {module}")
        except ImportError as e:
            missing_modules.append((module, str(e)))
            print(f"❌ {module}: {e}")

    print("\n📊 SUMMARY:")
    print(f"Available modules: {len(available_modules)}")
    print(f"Missing modules: {len(missing_modules)}")

    if missing_modules:
        print("\nMissing modules:")
        for module, error in missing_modules:
            print(f"  - {module}: {error}")

    return len(missing_modules) == 0

def test_basic_torch_export():
    """Test basic torch export functionality"""
    print("\n🧪 Testing Basic Torch Export...")

    try:
        import torch
        import torch.nn as nn

        # Simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        sample_input = torch.randn(1, 10)

        # Test basic export
        from torch.export import export
        exported_program = export(model, (sample_input,))
        print("✅ Basic torch.export works")

        # Test to_edge if available
        try:
            from executorch.exir import to_edge
            edge_program = to_edge(exported_program)
            print("✅ to_edge conversion works")
            return True
        except ImportError:
            print("⚠️  to_edge not available (ExecuTorch not fully built)")
            return False

    except Exception as e:
        print(f"❌ Torch export test failed: {e}")
        return False

def main():
    """Main diagnostic function"""
    executorch_ready = check_executorch_availability()
    torch_export_works = test_basic_torch_export()

    print("\n" + "=" * 50)
    print("🎯 DIAGNOSTIC RESULTS")
    print("=" * 50)

    if executorch_ready and torch_export_works:
        print("✅ ExecuTorch is fully available and working!")
        print("\n🚀 Ready to proceed with:")
        print("  - MaxPool2d XNNPack debugging")
        print("  - Backend partitioning tests")
        print("  - Performance analysis")
    elif torch_export_works:
        print("⚠️  Basic PyTorch export works, but ExecuTorch needs building")
        print("\n🔨 Next steps:")
        print("  1. Complete ExecuTorch build (may take time)")
        print("  2. Install missing dependencies")
        print("  3. Run full diagnostic suite")
    else:
        print("❌ Core functionality not available")
        print("\n🔧 Troubleshooting:")
        print("  1. Check Python environment (3.10-3.13)")
        print("  2. Verify PyTorch installation")
        print("  3. Re-run ExecuTorch build")
        print("  4. Check for missing system dependencies")

if __name__ == "__main__":
    main()