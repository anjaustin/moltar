#!/usr/bin/env python3
"""
Test XNNPack backend directly to see if it can lower MaxPool operations.
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
from executorch.exir.backend.backend_api import to_backend
from executorch.backends.xnnpack.xnnpack_preprocess import XnnpackBackend
from executorch.exir.backend.backend_details import CompileSpec

def test_xnnpack_backend_directly():
    """Test if XNNPack backend can lower a simple MaxPool operation"""
    print("🧪 TESTING XNNPACK BACKEND DIRECTLY")
    print("=" * 50)

    # Create simple MaxPool model
    class SimpleMaxPool(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool2d(2, 2)

        def forward(self, x):
            return self.pool(x)

    model = SimpleMaxPool()
    sample_input = torch.randn(1, 3, 32, 32)

    print("1. Exporting simple MaxPool model...")
    exported = export(model, (sample_input,))
    edge = to_edge(exported)
    cleaned_edge = run_lfn_xnnpack_pipeline(edge)

    method_name = list(cleaned_edge.methods)[0]
    exported_prog = cleaned_edge.exported_program(method_name)

    print("2. Checking operations...")
    ops = [node for node in exported_prog.graph_module.graph.nodes if node.op == 'call_function']
    maxpool_ops = [op for op in ops if 'max_pool' in str(op.target)]
    print(f"   Total ops: {len(ops)}")
    print(f"   MaxPool ops: {len(maxpool_ops)}")
    for op in maxpool_ops:
        print(f"      • {op.name}: {op.target}")

    print("\n3. Testing XNNPack backend directly...")
    try:
        compile_specs = [CompileSpec("target", "arm64-v8.2-a+dotprod")]
        lowered_module = to_backend(
            XnnpackBackend.__name__,
            exported_prog,
            compile_specs
        )
        print("   ✅ XNNPack backend lowered successfully!")
        print(f"   Backend ID: {lowered_module.backend_id}")
        print(f"   Compile specs: {lowered_module.compile_specs}")
        return True

    except Exception as e:
        print(f"   ❌ XNNPack backend failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_partitioned_submodule():
    """Test if XNNPack can lower a partitioned submodule"""
    print("\n🧪 TESTING PARTITIONED SUBMODULE LOWERING")
    print("=" * 50)

    # Create model and partition it
    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool2d(2, 2)

        def forward(self, x):
            return self.pool(x)

    model = TestModel()
    sample_input = torch.randn(1, 3, 32, 32)

    exported = export(model, (sample_input,))
    edge = to_edge(exported)
    cleaned_edge = run_lfn_xnnpack_pipeline(edge)

    # Partition
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    partitioner = XnnpackPartitioner()
    method_name = list(cleaned_edge.methods)[0]
    exported_prog = cleaned_edge.exported_program(method_name)

    partitioned = partitioner(exported_prog)

    if not hasattr(partitioned, 'tagged_exported_program'):
        print("❌ No tagged program")
        return False

    tagged_ep = partitioned.tagged_exported_program

    print("1. Finding tagged submodules...")
    submodules = {}
    for name, module in tagged_ep.graph_module.named_modules():
        if name.startswith('fused_'):
            submodules[name] = module
            print(f"   Found submodule: {name}")

    if not submodules:
        print("❌ No submodules found")
        return False

    print("\n2. Testing submodule lowering...")
    compile_specs = [CompileSpec("target", "arm64-v8.2-a+dotprod")]

    for name, submodule in submodules.items():
        print(f"\n   Testing submodule: {name}")
        print(f"   Operations in submodule: {len(list(submodule.graph.nodes))}")

        # Check if it contains MaxPool
        has_maxpool = any('max_pool' in str(node.target) for node in submodule.graph.nodes if node.op == 'call_function')
        print(f"   Contains MaxPool: {has_maxpool}")

        try:
            # Create an ExportedProgram from the submodule
            from torch.export import export
            submodule_input = torch.randn(1, 3, 32, 32)  # Same shape as original
            submodule_exported = export(submodule, (submodule_input,))

            lowered = to_backend(
                XnnpackBackend.__name__,
                submodule_exported,
                compile_specs
            )
            print(f"   ✅ Submodule {name} lowered successfully!")

        except Exception as e:
            print(f"   ❌ Submodule {name} lowering failed: {e}")
            return False

    print("\n✅ All submodules can be lowered individually!")
    return True

if __name__ == "__main__":
    print("Testing XNNPack backend capabilities...\n")

    test1 = test_xnnpack_backend_directly()
    test2 = test_partitioned_submodule()

    print("\n" + "=" * 70)
    print("🎯 XNNPACK BACKEND TEST RESULTS")
    print("=" * 70)
    print(f"Direct backend test: {'✅ PASSED' if test1 else '❌ FAILED'}")
    print(f"Submodule test: {'✅ PASSED' if test2 else '❌ FAILED'}")

    if test1 and test2:
        print("\n🎉 XNNPack backend works! The issue is in the delegation pipeline.")
        print("   The backend can lower MaxPool operations, but the to_backend()")
        print("   pipeline is not calling it on the partitioned submodules.")
    else:
        print("\n❌ XNNPack backend has issues with MaxPool operations.")
        print("   This explains why delegation fails.")