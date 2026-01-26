#!/usr/bin/env python3
"""
SpaceGhost: MaxPool2d Implementation Falsification

Scientific falsification testing of REQ-XNN-001 claims:
- MaxPool2d operations successfully partition to XNNPack
- No performance regressions introduced
- Proper constraint validation
- Indices output handling works correctly

Following the SpaceGhost falsification methodology:
1. State the hypothesis clearly
2. Design experiments to test it
3. Attempt to disprove the hypothesis
4. Accept or reject based on evidence
"""

import sys
import os
import time
import torch
import torch.nn as nn
from torch.export import export

# Add ExecuTorch to path
script_dir = os.path.dirname(os.path.abspath(__file__))
executorch_lib = os.path.join(script_dir, 'executorch', 'pip-out', 'lib.macosx-26.0-arm64-cpython-311')
sys.path.insert(0, executorch_lib)

from executorch.exir import to_edge
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

# Falsification test results
test_results = []

def log_test_result(test_name, hypothesis, result, evidence, conclusion):
    """Log a falsification test result"""
    test_results.append({
        'test': test_name,
        'hypothesis': hypothesis,
        'result': result,
        'evidence': evidence,
        'conclusion': conclusion
    })

    status = "✅ ACCEPTED" if result else "❌ FALSIFIED"
    print(f"\n🧪 {test_name}")
    print(f"   Hypothesis: {hypothesis}")
    print(f"   Result: {status}")
    print(f"   Evidence: {evidence}")
    print(f"   Conclusion: {conclusion}")

def test_maxpool_partitioning_success():
    """
    Test Hypothesis: MaxPool2d operations successfully partition to XNNPack backend
    """
    hypothesis = "MaxPool2d operations in CNN models can be successfully partitioned to XNNPack backend"

    try:
        # Test model with MaxPool2d
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

        # Export pipeline
        exported = export(model, (sample_input,))
        edge = to_edge(exported)

        # Partitioning
        partitioner = XnnpackPartitioner()
        method_name = list(edge.methods)[0]
        exported_prog = edge.exported_program(method_name)

        start_time = time.time()
        partitioned = partitioner(exported_prog)
        partition_time = time.time() - start_time

        # Check results
        if hasattr(partitioned, 'tagged_exported_program'):
            tagged_ep = partitioned.tagged_exported_program
            ops = []
            for node in tagged_ep.graph_module.graph.nodes:
                if node.op == 'call_function':
                    ops.append(str(node.target))

            maxpool_ops = [op for op in ops if 'max_pool' in op.lower()]
            has_maxpool_partitioned = len(maxpool_ops) == 0  # Should be removed from main graph

            evidence = f"Partitioning completed in {partition_time:.3f}s. MaxPool operations in main graph: {len(maxpool_ops)}"
            conclusion = "MaxPool2d successfully partitioned to XNNPack" if has_maxpool_partitioned else "MaxPool2d remained in main graph"

            log_test_result("MaxPool Partitioning", hypothesis, has_maxpool_partitioned, evidence, conclusion)
            return has_maxpool_partitioned

    except Exception as e:
        evidence = f"Partitioning failed with error: {e}"
        conclusion = "Implementation has critical bugs"
        log_test_result("MaxPool Partitioning", hypothesis, False, evidence, conclusion)
        return False

def test_indices_constraint_validation():
    """
    Test Hypothesis: MaxPool2d partitioning correctly validates indices output usage
    """
    hypothesis = "MaxPool2d partitioning rejects operations where indices output is used"

    try:
        # Test model that uses MaxPool2d indices (should fail partitioning)
        class IndicesUsingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 3, 1, 1)
                self.pool = nn.MaxPool2d(2, 2, return_indices=True)

            def forward(self, x):
                x = torch.relu(self.conv(x))
                pooled, indices = self.pool(x)
                # Use indices in some operation (this should prevent partitioning)
                result = torch.gather(x.view(x.size(0), -1), 1, indices.view(-1))
                return result

        model = IndicesUsingModel()
        sample_input = torch.randn(1, 3, 32, 32)

        # This should either fail partitioning or succeed but with indices in main graph
        exported = export(model, (sample_input,))
        edge = to_edge(exported)

        partitioner = XnnpackPartitioner()
        method_name = list(edge.methods)[0]
        exported_prog = edge.exported_program(method_name)

        partitioned = partitioner(exported_prog)

        # Check if partitioning succeeded (it shouldn't for indices-using model)
        partitioning_attempted = hasattr(partitioned, 'tagged_exported_program')

        if partitioning_attempted:
            # Check if MaxPool2d operations are still in main graph (they should be)
            tagged_ep = partitioned.tagged_exported_program
            ops = []
            for node in tagged_ep.graph_module.graph.nodes:
                if node.op == 'call_function':
                    ops.append(str(node.target))

            maxpool_ops = [op for op in ops if 'max_pool' in op.lower()]
            correctly_rejected = len(maxpool_ops) > 0  # Should still be in main graph

            evidence = f"Indices-using model partitioning: {'rejected' if correctly_rejected else 'incorrectly accepted'}"
            conclusion = "Indices constraint validation working" if correctly_rejected else "Indices constraint validation failed"

            log_test_result("Indices Constraint", hypothesis, correctly_rejected, evidence, conclusion)
            return correctly_rejected
        else:
            evidence = "Partitioning failed completely for indices-using model"
            conclusion = "Unable to test constraint validation"
            log_test_result("Indices Constraint", hypothesis, False, evidence, conclusion)
            return False

    except Exception as e:
        evidence = f"Indices constraint test failed: {e}"
        conclusion = "Test setup issue, not constraint validation problem"
        log_test_result("Indices Constraint", hypothesis, False, evidence, conclusion)
        return False

def test_stride_kernel_constraints():
    """
    Test Hypothesis: MaxPool2d partitioning enforces stride <= kernel_size constraints
    """
    hypothesis = "MaxPool2d partitioning rejects operations with invalid stride > kernel_size"

    try:
        # Test model with invalid stride (should fail partitioning constraints)
        class InvalidStrideModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 3, 1, 1)
                # Invalid: stride (4,4) > kernel_size (2,2)
                self.pool = nn.MaxPool2d(kernel_size=2, stride=4)

            def forward(self, x):
                x = torch.relu(self.conv(x))
                x = self.pool(x)
                return x

        model = InvalidStrideModel()
        sample_input = torch.randn(1, 3, 32, 32)

        exported = export(model, (sample_input,))
        edge = to_edge(exported)

        partitioner = XnnpackPartitioner()
        method_name = list(edge.methods)[0]
        exported_prog = edge.exported_program(method_name)

        partitioned = partitioner(exported_prog)

        # Check if MaxPool2d operations are still in main graph (they should be due to constraint failure)
        if hasattr(partitioned, 'tagged_exported_program'):
            tagged_ep = partitioned.tagged_exported_program
            ops = []
            for node in tagged_ep.graph_module.graph.nodes:
                if node.op == 'call_function':
                    ops.append(str(node.target))

            maxpool_ops = [op for op in ops if 'max_pool' in op.lower()]
            correctly_rejected = len(maxpool_ops) > 0  # Should still be in main graph

            evidence = f"Invalid stride model: {'correctly rejected' if correctly_rejected else 'incorrectly accepted'}"
            conclusion = "Stride constraint validation working" if correctly_rejected else "Stride constraint validation failed"

            log_test_result("Stride Constraints", hypothesis, correctly_rejected, evidence, conclusion)
            return correctly_rejected
        else:
            evidence = "Partitioning failed for invalid stride model"
            conclusion = "Unable to test stride constraints"
            log_test_result("Stride Constraints", hypothesis, False, evidence, conclusion)
            return False

    except Exception as e:
        evidence = f"Stride constraint test failed: {e}"
        conclusion = "Test setup issue"
        log_test_result("Stride Constraints", hypothesis, False, evidence, conclusion)
        return False

def test_no_performance_regression():
    """
    Test Hypothesis: MaxPool2d partitioning doesn't introduce performance regressions
    """
    hypothesis = "MaxPool2d partitioning doesn't significantly slow down the partitioning process"

    try:
        # Compare partitioning time with and without MaxPool2d
        class WithMaxPoolModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 3, 1, 1)
                self.pool = nn.MaxPool2d(2, 2)
                self.fc = nn.Linear(64 * 16 * 16, 10)

            def forward(self, x):
                x = torch.relu(self.conv(x))
                x = self.pool(x)
                return self.fc(x.view(x.size(0), -1))

        class WithoutMaxPoolModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 3, 1, 1)
                self.fc = nn.Linear(64 * 32 * 32, 10)  # No pooling, larger input

            def forward(self, x):
                x = torch.relu(self.conv(x))
                return self.fc(x.view(x.size(0), -1))

        # Test both models
        models = {
            'with_maxpool': WithMaxPoolModel(),
            'without_maxpool': WithoutMaxPoolModel()
        }

        times = {}

        for name, model in models.items():
            sample_input = torch.randn(1, 3, 32, 32)

            exported = export(model, (sample_input,))
            edge = to_edge(exported)

            partitioner = XnnpackPartitioner()
            method_name = list(edge.methods)[0]
            exported_prog = edge.exported_program(method_name)

            start_time = time.time()
            partitioned = partitioner(exported_prog)
            end_time = time.time()

            times[name] = end_time - start_time

        # Check for significant regression (arbitrary threshold: 10x slower)
        regression_ratio = times['with_maxpool'] / times['without_maxpool']
        no_regression = regression_ratio < 10.0  # Allow up to 10x slower, should be much less

        evidence = f"Partitioning times: With MaxPool {times['with_maxpool']:.3f}s, Without {times['without_maxpool']:.3f}s (ratio: {regression_ratio:.2f}x)"
        conclusion = "No significant performance regression" if no_regression else f"Performance regression detected ({regression_ratio:.1f}x slower)"

        log_test_result("Performance Regression", hypothesis, no_regression, evidence, conclusion)
        return no_regression

    except Exception as e:
        evidence = f"Performance regression test failed: {e}"
        conclusion = "Unable to measure performance impact"
        log_test_result("Performance Regression", hypothesis, False, evidence, conclusion)
        return False

def test_lfn_compatibility():
    """
    Test Hypothesis: MaxPool2d fix enables LFN-style models with pooling operations
    """
    hypothesis = "LFN-style models with MaxPool2d operations can now be deployed on XNNPack"

    try:
        # Create LFN-inspired model with MaxPool2d
        class LFNStyleModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Liquid-inspired projection
                self.input_proj = nn.Linear(784, 256)
                self.liquid_layer = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64)
                )
                # Convolutional processing (common in LFN variants)
                self.conv_proj = nn.Conv2d(1, 32, kernel_size=3, padding=1)
                self.maxpool = nn.MaxPool2d(2, 2)  # The critical operation
                self.output_proj = nn.Linear(64 + 32*14*14, 10)

            def forward(self, x):
                batch_size = x.size(0)
                x_flat = x.view(batch_size, -1)

                # Liquid processing
                liquid_out = self.liquid_layer(self.input_proj(x_flat))

                # Convolutional processing with MaxPool2d
                x_conv = x.view(batch_size, 1, 28, 28)  # MNIST-like
                x_conv = torch.relu(self.conv_proj(x_conv))
                x_conv = self.maxpool(x_conv)  # This was previously impossible
                x_conv = x_conv.view(batch_size, -1)

                # Combine and output
                combined = torch.cat([liquid_out, x_conv], dim=1)
                return self.output_proj(combined)

        model = LFNStyleModel()
        sample_input = torch.randn(1, 784)

        # Full pipeline test
        exported = export(model, (sample_input,))
        edge = to_edge(exported)

        partitioner = XnnpackPartitioner()
        method_name = list(edge.methods)[0]
        exported_prog = edge.exported_program(method_name)

        partitioned = partitioner(exported_prog)

        # Check if partitioning succeeded
        if hasattr(partitioned, 'tagged_exported_program'):
            tagged_ep = partitioned.tagged_exported_program
            ops = []
            for node in tagged_ep.graph_module.graph.nodes:
                if node.op == 'call_function':
                    ops.append(str(node.target))

            maxpool_ops = [op for op in ops if 'max_pool' in op.lower()]
            lfn_compatible = len(maxpool_ops) == 0  # MaxPool2d should be partitioned

            evidence = f"LFN-style model with MaxPool2d: {'successfully partitioned' if lfn_compatible else 'partitioning failed'}"
            conclusion = "LFN models with MaxPool2d now supported" if lfn_compatible else "LFN compatibility not achieved"

            log_test_result("LFN Compatibility", hypothesis, lfn_compatible, evidence, conclusion)
            return lfn_compatible
        else:
            evidence = "Partitioning failed for LFN-style model"
            conclusion = "LFN compatibility test failed"
            log_test_result("LFN Compatibility", hypothesis, False, evidence, conclusion)
            return False

    except Exception as e:
        evidence = f"LFN compatibility test failed: {e}"
        conclusion = "Test setup issue"
        log_test_result("LFN Compatibility", hypothesis, False, evidence, conclusion)
        return False

def run_falsification_suite():
    """Run the complete falsification test suite"""
    print("🔬 SpaceGhost: MaxPool2d Implementation Falsification")
    print("=" * 60)
    print("Testing REQ-XNN-001 claims using scientific falsification methodology")
    print("Each test attempts to disprove our implementation claims")
    print()

    # Run all tests
    tests = [
        test_maxpool_partitioning_success,
        test_indices_constraint_validation,
        test_stride_kernel_constraints,
        test_no_performance_regression,
        test_lfn_compatibility
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
        print()

    # Summary
    passed = sum(results)
    total = len(results)
    success_rate = passed / total * 100

    print("=" * 60)
    print("📊 FALSIFICATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"Tests Passed: {passed}/{total} ({success_rate:.1f}%)")

    if success_rate >= 80:
        print("✅ MAJORITY ACCEPTED: Implementation claims largely validated")
        print("   The MaxPool2d fix successfully addresses the identified issues")
    elif success_rate >= 60:
        print("⚠️  PARTIALLY ACCEPTED: Some claims validated, others need work")
        print("   Implementation has merit but requires refinement")
    else:
        print("❌ MOSTLY FALSIFIED: Implementation claims largely disproven")
        print("   Significant issues remain, major rework needed")

    print("\n📋 DETAILED RESULTS:")
    for i, result in enumerate(test_results):
        status = "✅ PASSED" if results[i] else "❌ FAILED"
        print(f"   {status}: {result['test']}")

    # Overall conclusion
    all_passed = all(results)
    if all_passed:
        print("\n🎉 CONCLUSION: REQ-XNN-001 implementation SUCCESSFULLY VALIDATED")
        print("   All falsification attempts failed - claims are supported by evidence!")
        print("   MaxPool2d operations now work correctly in XNNPack backend")
    else:
        print("\n⚠️  CONCLUSION: REQ-XNN-001 implementation PARTIALLY VALIDATED")
        print("   Some falsification attempts succeeded - claims need refinement")
        print("   Review failed tests and address identified issues")

    return all_passed

if __name__ == "__main__":
    success = run_falsification_suite()
    sys.exit(0 if success else 1)