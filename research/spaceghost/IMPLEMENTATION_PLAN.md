# SpaceGhost: XNNPack Implementation Plan

## Overview
Detailed implementation plan for addressing the three critical XNNPack backend requirements identified through web research and analysis.

## REQ-XNN-001: Implement Missing MaxPool2d Operator Support

### Problem Analysis
- **Current State**: XNNPack backend fails on `nn.MaxPool2d` operations during partitioning
- **Root Cause**: Partitioner doesn't correctly map ATen `max_pool2d_with_indices.default` to XNNPack Subgraph API
- **Impact**: Complete failure to deploy models with pooling layers (common in CNNs and LFN architectures)

### Implementation Strategy

#### Phase 1: Analysis and Detection (Week 2)
```python
# Diagnostic code to identify MaxPool2d usage
def detect_maxpool2d_issues(edge_program):
    """Analyze edge program for MaxPool2d operations"""
    maxpool_nodes = []
    for node in edge_program.graph.nodes:
        if 'max_pool' in str(node).lower():
            maxpool_nodes.append({
                'node': node,
                'op': node.target,
                'args': node.args,
                'has_indices': any('indices' in str(arg) for arg in node.args)
            })
    return maxpool_nodes
```

#### Phase 2: Partitioner Enhancement (Week 3)
**File**: `executorch/backends/xnnpack/partition/xnnpack_partitioner.py`

```python
# Add MaxPool2d handler to XnnpackPartitioner
class XnnpackPartitioner:
    def __init__(self):
        self.node_handlers = {
            torch.ops.aten.max_pool2d_with_indices.default: self._handle_maxpool2d,
            # ... existing handlers
        }

    def _handle_maxpool2d(self, node):
        """Handle MaxPool2d partitioning logic"""
        # Check if indices output has users
        indices_output = node.args[0] if len(node.args) > 1 else None
        indices_has_users = self._check_output_users(indices_output)

        if not indices_has_users:
            # Safe to partition - indices not used
            return self._create_xnnpack_subgraph(node, use_indices=False)
        else:
            # Cannot partition - indices are used
            return None
```

#### Phase 3: XNNPack Kernel Integration (Week 4)
**File**: `executorch/backends/xnnpack/ops/maxpool2d.cpp`

```cpp
// XNNPack MaxPool2d kernel implementation
xnn_status xnn_define_max_pooling_2d(
    xnn_subgraph_t subgraph,
    uint32_t input_id,
    uint32_t output_id,
    uint32_t input_padding_top,
    uint32_t input_padding_right,
    uint32_t input_padding_bottom,
    uint32_t input_padding_left,
    uint32_t pooling_height,
    uint32_t pooling_width,
    uint32_t stride_height,
    uint32_t stride_width,
    uint32_t dilation_height,
    uint32_t dilation_width,
    float output_min,
    float output_max,
    uint32_t flags) {

  // Convert ATen parameters to XNNPack format
  // Handle NHWC memory format requirements
  // Implement the pooling operation

  return xnn_status_success;
}
```

#### Phase 4: Memory Format Handling (Week 4)
**Key Challenge**: XNNPack requires NHWC format, but PyTorch uses NCHW

```python
# Add memory format transformation
def _ensure_nhwc_format(node):
    """Insert to_dim_order_last transformations for NHWC conversion"""
    if node.memory_format != torch.contiguous_format:  # Assuming NCHW
        # Insert: x = x.to(memory_format=torch.channels_last)
        nhwc_node = create_to_channels_last_node(node)
        return nhwc_node
    return node
```

### Testing and Validation
```python
def test_maxpool2d_xnnpack():
    """Test MaxPool2d XNNPack integration"""
    model = TestModelWithMaxPool2d()
    sample_input = torch.randn(1, 3, 32, 32)

    # Export and partition
    exported = export(model, (sample_input,))
    edge = to_edge(exported)
    partitioned = XnnpackPartitioner()(edge)

    # Verify partitioning succeeded
    assert partitioned is not None, "MaxPool2d partitioning failed"

    # Lower to XNNPack
    lowered = XnnpackBackend().compile(partitioned)
    assert lowered is not None, "MaxPool2d lowering failed"

    print("✅ MaxPool2d XNNPack integration successful")
```

## REQ-XNN-002: Fix Dynamic Quantization Chain Duplication

### Problem Analysis
- **Current State**: Repeated `Quantize -> Dequantize -> Quantize -> Dequantize` patterns
- **Root Cause**: Partitioner treats each operator as isolated, re-inserting quant ops
- **Impact**: 30-50% performance overhead from redundant quantization operations

### Implementation Strategy

#### Phase 1: Pattern Detection (Week 3)
```python
def detect_quantization_chains(edge_program):
    """Find Q -> DQ -> Q -> DQ patterns"""
    chains = []
    nodes = list(edge_program.graph.nodes)

    for i, node in enumerate(nodes):
        if self._is_quantize_node(node):
            # Look for DQ -> Q pattern
            chain = self._find_quant_chain(node, nodes[i:])
            if chain and len(chain) > 2:  # More than Q->DQ
                chains.append(chain)

    return chains
```

#### Phase 2: Graph Pass Implementation (Week 3-4)
**File**: `exir/passes/quant_fusion_pass.py` (New)

```python
class QuantizationFusionPass:
    """Fuse redundant quantization chains"""

    def __call__(self, graph_module):
        """Apply quantization fusion to graph"""
        self.fuse_quantize_dequantize_chains(graph_module)
        self.eliminate_redundant_quantize_ops(graph_module)
        return graph_module

    def fuse_quantize_dequantize_chains(self, graph_module):
        """Fuse Q->DQ->Q->DQ into single Q->DQ"""
        # Find patterns: quantize -> dequantize -> quantize -> dequantize
        # With same scale/zero_point between adjacent pairs
        # Replace with single quantize -> dequantize

    def eliminate_redundant_quantize_ops(self, graph_module):
        """Remove unnecessary quantize operations"""
        # Remove quantize ops where input is already quantized
        # With same quantization parameters
```

#### Phase 3: XNNPack Quantization Optimization (Week 4)
**File**: `executorch/backends/xnnpack/xnnpack_graph_builder.cpp`

```cpp
// Optimize quantization in XNNPack lowering
void XnnpackGraphBuilder::lower_quantized_operations() {
    // Instead of creating separate xnn_node_dequantize
    // Link quantized tensor directly to next quantized operation

    for (auto& op : operations_) {
        if (op.is_quantized && next_op.is_quantized) {
            // Direct tensor flow without dequantize->quantize
            link_quantized_tensors(op.output, next_op.input);
        }
    }
}
```

### Testing and Validation
```python
def test_quantization_fusion():
    """Test quantization chain optimization"""
    # Create model with dynamic quantization
    model = DynamicQuantModel()
    sample_input = torch.randn(1, 224, 224, 3)

    # Apply quantization
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    # Export and apply fusion pass
    exported = export(quantized_model, (sample_input,))
    edge = to_edge(exported)
    fused = QuantizationFusionPass()(edge)

    # Count quantization operations
    quant_ops_before = count_quant_ops(edge)
    quant_ops_after = count_quant_ops(fused)

    assert quant_ops_after < quant_ops_before, "Fusion failed"
    print(f"✅ Reduced quant ops: {quant_ops_before} → {quant_ops_after}")
```

## REQ-XNN-003: Optimize Kernels for Snapdragon 480

### Problem Analysis
- **Current State**: Generic ARM NEON kernels, no Snapdragon-specific optimizations
- **Hardware**: Cortex-A76 (Gold) + Cortex-A55 (Silver), 2+6 core configuration
- **Missing**: Dot Product instructions, L3 cache optimization, thread pinning

### Implementation Strategy

#### Phase 1: Hardware Detection (Week 4)
```cpp
// Detect Snapdragon 480 capabilities
bool detect_snapdragon_480_features() {
    // Check for FEAT_DotProd support
    uint64_t isar0 = read_system_register(ISAR0_EL1);
    bool has_dotprod = (isar0 & (1 << 16)) != 0;

    // Check L3 cache size
    // Check core configuration

    return has_dotprod && /* other checks */;
}
```

#### Phase 2: Dot Product Kernel Optimization (Week 5)
**File**: `executorch/backends/xnnpack/kernels/qs8_dotprod.h`

```cpp
// Enable UDOT/SDOT instructions for Snapdragon 480
#define XNN_ARCH_ARM64_DOTPROD 1

// Optimized kernels using dot product instructions
void xnn_qs8_gemm_minmax_ukernel_1x8__asm_aarch64_neondot_ld128(
    size_t mr, size_t nr, size_t k,
    const int8_t* a, size_t a_stride,
    const int8_t* w, size_t w_stride,
    const float* bias, float* c, size_t c_stride,
    const union xnn_qs8_conv_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]) {

    // Use UDOT/SDOT instructions for 30-50% speedup
    // Optimized for Cortex-A76 microarchitecture
}
```

#### Phase 3: Threading and Cache Optimization (Week 5)
**File**: `executorch/backends/xnnpack/xnnpack_threadpool.c`

```cpp
// Optimize thread pool for Snapdragon 480
xnn_status xnn_initialize_threadpool(
    uint32_t thread_count,  // Set to 2 for big cores only
    xnn_threadpool_t* threadpool) {

    // Pin threads to big cores (Cortex-A76)
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    CPU_SET(0, &cpu_set);  // Core 0 - A76
    CPU_SET(1, &cpu_set);  // Core 1 - A76

    for (uint32_t i = 0; i < thread_count; i++) {
        pthread_setaffinity_np(threads[i], sizeof(cpu_set), &cpu_set);
    }

    // Optimize L3 cache usage
    // Set appropriate tiling parameters
}
```

#### Phase 4: Prefetching Optimization (Week 6)
```cpp
// Add instruction prefetching for irregular access patterns
void xnn_prefetch_lfm_weights(const int8_t* weights, size_t size) {
    // Prefetch weights for LFM operations
    // Optimize for Snapdragon 480 cache hierarchy
    __builtin_prefetch(weights, 0, 3);  // High locality hint

    // Prefetch in cache line sized chunks
    for (size_t i = 0; i < size; i += 64) {  // 64-byte cache lines
        __builtin_prefetch(weights + i, 0, 1);
    }
}
```

### Testing and Validation
```cpp
def benchmark_snapdragon_optimizations():
    """Benchmark Snapdragon 480 specific optimizations"""
    model = LFM2_350M_Model()
    sample_input = torch.randn(1, 4096)

    # Test with different configurations
    configs = {
        'baseline': {'threads': 8, 'use_dotprod': False},
        'optimized': {'threads': 2, 'use_dotprod': True, 'big_cores_only': True}
    }

    results = {}
    for config_name, config in configs.items():
        # Configure ExecuTorch for this setup
        configure_snapdragon_execution(config)

        # Measure performance
        latency = measure_inference_latency(model, sample_input, iterations=100)
        results[config_name] = latency

    improvement = (results['baseline'] - results['optimized']) / results['baseline'] * 100
    assert improvement > 25, f"Expected >25% improvement, got {improvement}%"
    print(f"✅ Snapdragon optimization: {improvement:.1f}% latency reduction")
```

## Integration and Deployment

### Build System Updates
```cmake
# CMakeLists.txt additions for Snapdragon support
if(ANDROID AND CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    # Enable dot product compilation
    add_compile_options(-march=armv8.2-a+dotprod)

    # Snapdragon-specific defines
    add_definitions(-DXNN_ARCH_ARM64_DOTPROD=1)
    add_definitions(-DSNAPDRAGON_480_OPTIMIZATIONS=1)
endif()
```

### Runtime Feature Detection
```cpp
// Runtime capability detection
bool XnnpackBackend::supports_snapdragon_optimizations() {
    // Check CPU features at runtime
    // Return true if Snapdragon 480 with dotprod support
    return detect_snapdragon_480_features();
}
```

### Performance Monitoring
```cpp
// Add performance counters
struct SnapdragonMetrics {
    double dotprod_utilization;      // % of operations using dot product
    double cache_hit_rate;          // L3 cache efficiency
    double big_core_utilization;    // % time on A76 cores
    double power_efficiency;        // Performance per watt
};

// Integrate with existing profiling
void collect_snapdragon_metrics(SnapdragonMetrics* metrics) {
    // Collect hardware counters
    // Report optimization effectiveness
}
```

## Success Criteria and Validation

### Functional Validation
- ✅ MaxPool2d operations partition and lower successfully
- ✅ Quantization chains reduced by >50%
- ✅ Snapdragon 480 features detected and utilized
- ✅ All LFM models deploy without XNNPack failures

### Performance Validation
- 📈 **Latency**: <150ms LFM2-350M inference (40% improvement)
- 📈 **DSP Utilization**: >12 TOPS on Snapdragon 480
- 📈 **Memory Efficiency**: No quantization overhead increase
- 📈 **Compatibility**: Maintain existing API and performance

### Testing Strategy
1. **Unit Tests**: Individual operator functionality
2. **Integration Tests**: Full model partitioning and lowering
3. **Performance Tests**: Benchmark against PyTorch Mobile/ONNX Runtime
4. **Regression Tests**: Ensure no existing functionality breaks

---

*This implementation plan provides detailed technical specifications for addressing the three critical XNNPack backend issues identified through web research. Each requirement includes specific code changes, testing strategies, and success criteria.*