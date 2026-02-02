#!/usr/bin/env python3
"""
ExecuTorch Quantization Metadata Format

Defines the metadata format for quantized models in ExecuTorch,
enabling proper loading and execution of block-wise quantized tensors.
"""

import torch
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from quantization_utils import QuantizedTensor

@dataclass
class QuantizedLayerMetadata:
    """Metadata for a single quantized layer"""
    name: str
    original_shape: Tuple[int, ...]
    quantized_shape: Tuple[int, ...]
    bits: int
    block_size: int
    symmetric: bool
    scales_shape: Tuple[int, ...]
    zeros_shape: Optional[Tuple[int, ...]]
    data_type: str  # "uint8", "int8", etc.
    compression_ratio: float
    original_size_bytes: int
    quantized_size_bytes: int

@dataclass
class QuantizationModelMetadata:
    """Complete metadata for a quantized model"""
    model_name: str
    original_model_name: str
    quantization_config: Dict[str, Any]
    layers: List[QuantizedLayerMetadata]
    total_original_size_bytes: int
    total_quantized_size_bytes: int
    overall_compression_ratio: float
    calibration_samples_used: int
    accuracy_metrics: Dict[str, float]
    creation_timestamp: float
    execuTorch_version: str
    target_platform: str  # "mobile", "server", etc.

class ExecuTorchQuantizationSerializer:
    """
    Serializer for quantized models compatible with ExecuTorch

    Handles:
    - Quantized tensor serialization
    - Metadata generation
    - ExecuTorch-compatible format
    - Mobile deployment optimization
    """

    def __init__(self, target_platform: str = "mobile"):
        self.target_platform = target_platform

    def serialize_quantized_model(self,
                                quantized_layers: Dict[str, QuantizedTensor],
                                model_name: str,
                                output_dir: str,
                                accuracy_metrics: Optional[Dict[str, float]] = None,
                                calibration_samples: int = 0) -> str:
        """
        Serialize quantized model in ExecuTorch-compatible format

        Args:
            quantized_layers: Dictionary of quantized layers
            model_name: Name for the quantized model
            output_dir: Output directory
            accuracy_metrics: Optional accuracy validation results
            calibration_samples: Number of calibration samples used

        Returns:
            Path to the serialized model
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Serializing quantized model '{model_name}' for ExecuTorch...")

        # Create layer metadata
        layer_metadata = []
        total_original_size = 0
        total_quantized_size = 0

        for layer_name, quantized in quantized_layers.items():
            # Calculate sizes
            original_size = int(torch.prod(torch.tensor(quantized.original_shape))) * 4  # Assume FP32
            quantized_size = self._calculate_quantized_size_bytes(quantized)
            compression_ratio = original_size / quantized_size if quantized_size > 0 else 1.0

            total_original_size += original_size
            total_quantized_size += quantized_size

            # Create layer metadata
            layer_meta = QuantizedLayerMetadata(
                name=layer_name,
                original_shape=quantized.original_shape,
                quantized_shape=quantized.data.shape,
                bits=quantized.bits,
                block_size=quantized.block_size,
                symmetric=quantized.zeros is None,
                scales_shape=quantized.scales.shape,
                zeros_shape=quantized.zeros.shape if quantized.zeros is not None else None,
                data_type=self._get_data_type_string(quantized.data.dtype),
                compression_ratio=compression_ratio,
                original_size_bytes=original_size,
                quantized_size_bytes=quantized_size
            )
            layer_metadata.append(layer_meta)

            # Save quantized tensor
            self._save_quantized_tensor(quantized, layer_name, output_path)

        # Create model metadata
        overall_ratio = total_original_size / total_quantized_size if total_quantized_size > 0 else 1.0

        model_metadata = QuantizationModelMetadata(
            model_name=model_name,
            original_model_name="LFM2-350M",  # Would be dynamic
            quantization_config={
                'bits': 4,  # Would be from config
                'block_size': 64,
                'symmetric': True,
                'target_platform': self.target_platform
            },
            layers=layer_metadata,
            total_original_size_bytes=total_original_size,
            total_quantized_size_bytes=total_quantized_size,
            overall_compression_ratio=overall_ratio,
            calibration_samples_used=calibration_samples,
            accuracy_metrics=accuracy_metrics or {},
            creation_timestamp=torch.cuda.FloatTensor(1).new_tensor(torch.tensor(time.time())).item() if torch.cuda.is_available() else time.time(),
            execuTorch_version="0.1.0",  # Would be dynamic
            target_platform=self.target_platform
        )

        # Save metadata
        metadata_path = output_path / "quantization_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(model_metadata), f, indent=2, default=str)

        print("Quantized model serialization complete:"        print(f"  Layers: {len(layer_metadata)}")
        print(f"  Original size: {total_original_size / (1024*1024):.1f}MB")
        print(f"  Quantized size: {total_quantized_size / (1024*1024):.1f}MB")
        print(f"  Compression ratio: {overall_ratio:.1f}x")
        print(f"  Saved to: {output_path}")

        return str(output_path)

    def _calculate_quantized_size_bytes(self, quantized: QuantizedTensor) -> int:
        """Calculate the total size of a quantized tensor in bytes"""
        data_size = quantized.data.numel() * quantized.data.element_size()
        scales_size = quantized.scales.numel() * quantized.scales.element_size()
        zeros_size = quantized.zeros.numel() * quantized.zeros.element_size() if quantized.zeros is not None else 0

        # Add metadata overhead (minimal)
        metadata_size = 64  # Rough estimate for metadata

        return data_size + scales_size + zeros_size + metadata_size

    def _get_data_type_string(self, dtype: torch.dtype) -> str:
        """Convert torch dtype to string"""
        dtype_map = {
            torch.uint8: "uint8",
            torch.int8: "int8",
            torch.int16: "int16",
            torch.int32: "int32",
            torch.float16: "float16",
            torch.float32: "float32"
        }
        return dtype_map.get(dtype, "unknown")

    def _save_quantized_tensor(self, quantized: QuantizedTensor,
                             layer_name: str, output_path: Path):
        """Save a quantized tensor to disk"""
        # Create safe filename
        safe_name = layer_name.replace('.', '_').replace('/', '_')
        tensor_path = output_path / f"{safe_name}.pt"

        # Save tensor data
        torch.save({
            'data': quantized.data,
            'scales': quantized.scales,
            'zeros': quantized.zeros,
            'bits': quantized.bits,
            'block_size': quantized.block_size,
            'original_shape': quantized.original_shape,
            'layer_name': layer_name
        }, tensor_path)

class ExecuTorchQuantizationLoader:
    """
    Loader for quantized models in ExecuTorch runtime

    Handles:
    - Loading quantized tensors from disk
    - Runtime dequantization
    - Memory-efficient loading
    - Hardware-specific optimizations
    """

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.metadata = None
        self.quantized_layers = {}

    def load_model_metadata(self) -> QuantizationModelMetadata:
        """Load model metadata"""
        metadata_path = self.model_path / "quantization_metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)

        # Convert back to dataclass
        self.metadata = QuantizationModelMetadata(**metadata_dict)
        return self.metadata

    def load_quantized_layer(self, layer_name: str) -> QuantizedTensor:
        """
        Load a quantized layer from disk

        Args:
            layer_name: Name of the layer to load

        Returns:
            QuantizedTensor for the layer
        """
        if layer_name in self.quantized_layers:
            return self.quantized_layers[layer_name]

        # Create safe filename
        safe_name = layer_name.replace('.', '_').replace('/', '_')
        tensor_path = self.model_path / f"{safe_name}.pt"

        if not tensor_path.exists():
            raise FileNotFoundError(f"Quantized tensor not found: {tensor_path}")

        # Load tensor data
        tensor_data = torch.load(tensor_path, map_location='cpu')

        # Reconstruct QuantizedTensor
        quantized = QuantizedTensor(
            data=tensor_data['data'],
            scales=tensor_data['scales'],
            zeros=tensor_data.get('zeros'),
            bits=tensor_data['bits'],
            block_size=tensor_data['block_size'],
            original_shape=tuple(tensor_data['original_shape'])
        )

        self.quantized_layers[layer_name] = quantized
        return quantized

    def preload_critical_layers(self, layer_names: List[str]):
        """
        Preload critical layers into memory for faster access

        Args:
            layer_names: List of layer names to preload
        """
        print(f"Preloading {len(layer_names)} critical layers...")

        for layer_name in layer_names:
            try:
                self.load_quantized_layer(layer_name)
            except Exception as e:
                print(f"Warning: Failed to preload layer {layer_name}: {e}")

        print("Critical layer preloading complete")

    def get_memory_usage_estimate(self) -> Dict[str, int]:
        """
        Estimate memory usage for loading the model

        Returns:
            Dictionary with memory estimates
        """
        if not self.metadata:
            self.load_model_metadata()

        return {
            'total_quantized_size_bytes': self.metadata.total_quantized_size_bytes,
            'total_original_size_bytes': self.metadata.total_original_size_bytes,
            'compression_ratio': self.metadata.overall_compression_ratio,
            'estimated_runtime_memory_mb': self.metadata.total_quantized_size_bytes * 1.5 / (1024 * 1024),  # 1.5x for overhead
            'layer_count': len(self.metadata.layers)
        }

class MobileOptimizedLoader(ExecuTorchQuantizationLoader):
    """
    Mobile-optimized loader for quantized models

    Optimizations:
    - ION memory mapping for zero-copy loading
    - Mali GPU memory layout optimizations
    - Progressive loading for large models
    """

    def __init__(self, model_path: str, use_ion_memory: bool = True):
        super().__init__(model_path)
        self.use_ion_memory = use_ion_memory

    def load_layer_mobile_optimized(self, layer_name: str) -> QuantizedTensor:
        """
        Load layer with mobile-specific optimizations

        Args:
            layer_name: Layer name to load

        Returns:
            Mobile-optimized QuantizedTensor
        """
        quantized = self.load_quantized_layer(layer_name)

        if self.use_ion_memory:
            # Apply ION memory optimizations (would integrate with actual ION allocator)
            quantized = self._optimize_for_ion_memory(quantized)

        # Apply Mali GPU memory layout optimizations
        quantized = self._optimize_for_mali_gpu(quantized)

        return quantized

    def _optimize_for_ion_memory(self, quantized: QuantizedTensor) -> QuantizedTensor:
        """Apply ION memory optimizations (placeholder)"""
        # In actual implementation, this would:
        # - Allocate ION coherent memory
        # - Copy data to ION buffers
        # - Set up Vulkan memory import
        return quantized

    def _optimize_for_mali_gpu(self, quantized: QuantizedTensor) -> QuantizedTensor:
        """Apply Mali GPU memory layout optimizations (placeholder)"""
        # In actual implementation, this would:
        # - Reorder data for optimal Mali access patterns
        # - Align to Mali cache line boundaries
        # - Optimize for Mali's memory hierarchy
        return quantized

def create_mobile_quantization_package(model_name: str,
                                     quantized_layers: Dict[str, QuantizedTensor],
                                     output_dir: str,
                                     accuracy_metrics: Optional[Dict[str, float]] = None) -> str:
    """
    Create a complete mobile quantization package

    Args:
        model_name: Name of the model
        quantized_layers: Quantized model layers
        output_dir: Output directory
        accuracy_metrics: Optional accuracy validation results

    Returns:
        Path to the created package
    """
    print(f"Creating mobile quantization package for {model_name}...")

    # Create serializer
    serializer = ExecuTorchQuantizationSerializer(target_platform="mobile")

    # Serialize model
    package_path = serializer.serialize_quantized_model(
        quantized_layers=quantized_layers,
        model_name=model_name,
        output_dir=output_dir,
        accuracy_metrics=accuracy_metrics,
        calibration_samples=100  # Would be dynamic
    )

    # Create mobile deployment script
    create_mobile_deployment_script(package_path)

    print(f"Mobile quantization package created at: {package_path}")
    return package_path

def create_mobile_deployment_script(package_path: str):
    """Create deployment script for mobile devices"""
    script_path = Path(package_path) / "deploy_mobile.sh"

    script_content = f'''#!/bin/bash
# Mobile Deployment Script for Quantized LFM Model
# Generated for Neural Interposer Integration

echo "Deploying quantized LFM model to mobile device..."

# Package path: {package_path}

# Copy model files to device
adb push {package_path}/*.pt /data/local/tmp/quantized_lfm/
adb push {package_path}/quantization_metadata.json /data/local/tmp/quantized_lfm/

# Set environment variables
export NI_QUANTIZED_MODEL_PATH="/data/local/tmp/quantized_lfm"
export NI_QUANTIZATION_BITS="4"
export NI_BLOCK_SIZE="64"

echo "Quantized model deployed successfully!"
echo "Ready for Neural Interposer execution"
'''

    with open(script_path, 'w') as f:
        f.write(script_content)

    # Make executable
    script_path.chmod(0o755)

    print(f"Mobile deployment script created: {script_path}")

if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 2:
        model_name = sys.argv[1]
        output_dir = sys.argv[2]
    else:
        # Demo
        print("Creating demo quantization package...")

        # Create dummy quantized layers
        dummy_layer = QuantizedTensor(
            data=torch.randint(0, 256, (100,), dtype=torch.uint8),
            scales=torch.randn(10),
            zeros=None,
            bits=4,
            block_size=64,
            original_shape=(1, 256)
        )

        quantized_layers = {
            'layer1.weight': dummy_layer,
            'layer2.weight': dummy_layer
        }

        package_path = create_mobile_quantization_package(
            model_name="LFM2-350M-Quantized",
            quantized_layers=quantized_layers,
            output_dir="demo_quantized_model",
            accuracy_metrics={'perplexity_ratio': 1.05, 'accuracy_drop': 0.02}
        )

        print(f"Demo package created: {package_path}")