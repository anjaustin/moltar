#!/usr/bin/env python3
"""
Block-wise Quantization Utilities for LFM Neural Interposer

Implements AirLLM-inspired quantization techniques optimized for:
- LFM2 model architectures
- Neural Interposer hardware acceleration
- ExecuTorch compatibility
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import os
from pathlib import Path

@dataclass
class QuantizationConfig:
    """Configuration for block-wise quantization"""
    bits: int = 4  # 4bit or 8bit quantization
    block_size: int = 64  # Block size for quantization
    symmetric: bool = True  # Symmetric quantization
    per_channel: bool = False  # Per-channel vs per-tensor scaling
    preserve_sparsity: bool = False  # Preserve zero weights
    quantize_attention: bool = True  # Quantize attention layers
    quantize_ffn: bool = True  # Quantize FFN layers
    quantize_embeddings: bool = False  # Usually keep embeddings in FP32

@dataclass
class QuantizedTensor:
    """Container for quantized tensor data"""
    data: torch.Tensor  # Quantized weights (packed)
    scales: torch.Tensor  # Dequantization scales
    zeros: Optional[torch.Tensor] = None  # Zero points for asymmetric quantization
    bits: int = 4
    block_size: int = 64
    original_shape: Tuple[int, ...] = None

class BlockwiseQuantizer:
    """AirLLM-inspired block-wise quantizer for LFM models"""

    def __init__(self, config: QuantizationConfig):
        self.config = config

    def quantize_tensor(self, tensor: torch.Tensor) -> QuantizedTensor:
        """
        Quantize a tensor using block-wise quantization

        Args:
            tensor: Input tensor to quantize

        Returns:
            QuantizedTensor with packed data and metadata
        """
        original_shape = tensor.shape
        tensor_flat = tensor.flatten()

        # Reshape into blocks
        num_blocks = (tensor_flat.numel() + self.config.block_size - 1) // self.config.block_size
        padded_size = num_blocks * self.config.block_size
        tensor_padded = torch.zeros(padded_size, dtype=tensor.dtype, device=tensor.device)
        tensor_padded[:tensor_flat.numel()] = tensor_flat

        # Reshape to [num_blocks, block_size]
        tensor_blocks = tensor_padded.view(num_blocks, self.config.block_size)

        # Compute scales and zeros for each block
        if self.config.symmetric:
            # Symmetric quantization: find max absolute value per block
            scales = tensor_blocks.abs().max(dim=1, keepdim=True)[0]
            scales = scales / (2**(self.config.bits - 1) - 1)
            scales = torch.clamp(scales, min=1e-8)  # Avoid division by zero
            zeros = None
        else:
            # Asymmetric quantization: find min/max per block
            block_min = tensor_blocks.min(dim=1, keepdim=True)[0]
            block_max = tensor_blocks.max(dim=1, keepdim=True)[0]
            scales = (block_max - block_min) / (2**self.config.bits - 1)
            scales = torch.clamp(scales, min=1e-8)
            zeros = -block_min / scales

        # Quantize each block
        if self.config.symmetric:
            quantized_blocks = torch.round(tensor_blocks / scales).clamp(
                -(2**(self.config.bits - 1)), 2**(self.config.bits - 1) - 1
            ).to(torch.int8)
        else:
            quantized_blocks = torch.round((tensor_blocks + zeros) / scales).clamp(
                0, 2**self.config.bits - 1
            ).to(torch.uint8)

        # Pack quantized data efficiently
        if self.config.bits == 4:
            # Pack two 4-bit values into one byte
            packed_data = self._pack_4bit_to_8bit(quantized_blocks)
        elif self.config.bits == 8:
            packed_data = quantized_blocks.to(torch.uint8)
        else:
            raise ValueError(f"Unsupported quantization bits: {self.config.bits}")

        return QuantizedTensor(
            data=packed_data,
            scales=scales.flatten(),
            zeros=zeros.flatten() if zeros is not None else None,
            bits=self.config.bits,
            block_size=self.config.block_size,
            original_shape=original_shape
        )

    def dequantize_tensor(self, quantized: QuantizedTensor) -> torch.Tensor:
        """
        Dequantize a QuantizedTensor back to full precision

        Args:
            quantized: Quantized tensor to dequantize

        Returns:
            Full precision tensor
        """
        # Unpack quantized data
        if quantized.bits == 4:
            unpacked_data = self._unpack_4bit_from_8bit(quantized.data)
        elif quantized.bits == 8:
            unpacked_data = quantized.data.to(torch.float32)
        else:
            raise ValueError(f"Unsupported quantization bits: {quantized.bits}")

        # Reshape to blocks
        num_blocks = len(quantized.scales)
        block_size = quantized.block_size
        unpacked_blocks = unpacked_data.view(num_blocks, block_size)

        # Apply dequantization
        if quantized.zeros is not None:
            # Asymmetric dequantization
            dequantized_blocks = unpacked_blocks * quantized.scales.unsqueeze(1) - quantized.zeros.unsqueeze(1)
        else:
            # Symmetric dequantization
            dequantized_blocks = unpacked_blocks * quantized.scales.unsqueeze(1)

        # Reshape back to original tensor
        tensor_padded = dequantized_blocks.flatten()
        tensor_size = int(np.prod(quantized.original_shape))
        tensor_flat = tensor_padded[:tensor_size]

        return tensor_flat.view(quantized.original_shape)

    def _pack_4bit_to_8bit(self, quantized: torch.Tensor) -> torch.Tensor:
        """Pack two 4-bit values into one byte"""
        # Convert to uint8 and ensure range 0-15
        quantized = quantized.to(torch.uint8) & 0x0F

        # Pack pairs of 4-bit values
        packed = torch.zeros((quantized.numel() + 1) // 2, dtype=torch.uint8, device=quantized.device)

        flat_quantized = quantized.flatten()
        for i in range(0, len(flat_quantized), 2):
            if i + 1 < len(flat_quantized):
                packed[i // 2] = (flat_quantized[i] << 4) | flat_quantized[i + 1]
            else:
                packed[i // 2] = flat_quantized[i] << 4

        return packed

    def _unpack_4bit_from_8bit(self, packed: torch.Tensor) -> torch.Tensor:
        """Unpack 4-bit values from bytes"""
        unpacked = torch.zeros(packed.numel() * 2, dtype=torch.float32, device=packed.device)

        for i, byte in enumerate(packed):
            # Extract high 4 bits
            unpacked[i * 2] = (byte >> 4) & 0x0F
            # Extract low 4 bits
            unpacked[i * 2 + 1] = byte & 0x0F

        return unpacked

class LFMQuantizer:
    """Quantizer specifically optimized for LFM2 models"""

    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.block_quantizer = BlockwiseQuantizer(config)

    def quantize_model(self, model: nn.Module) -> Dict[str, QuantizedTensor]:
        """
        Quantize an LFM2 model using layer-aware quantization

        Args:
            model: PyTorch LFM2 model to quantize

        Returns:
            Dictionary mapping layer names to quantized tensors
        """
        quantized_layers = {}

        for name, module in model.named_modules():
            if not self._should_quantize_layer(name, module):
                continue

            for param_name, param in module.named_parameters():
                if param_name in ['weight', 'bias']:
                    layer_name = f"{name}.{param_name}"
                    print(f"Quantizing {layer_name}: {param.shape}")

                    quantized = self.block_quantizer.quantize_tensor(param.data)
                    quantized_layers[layer_name] = quantized

                    # Calculate compression ratio
                    original_size = param.numel() * param.element_size()
                    quantized_size = self._calculate_quantized_size(quantized)
                    ratio = original_size / quantized_size

                    print(f"    Compression: {ratio:.1f}x")
        return quantized_layers

    def _should_quantize_layer(self, name: str, module: nn.Module) -> bool:
        """Determine if a layer should be quantized based on config"""
        # Skip embeddings if configured
        if not self.config.quantize_embeddings and 'embed' in name.lower():
            return False

        # Skip attention layers if configured
        if not self.config.quantize_attention and 'attention' in name.lower():
            return False

        # Skip FFN layers if configured
        if not self.config.quantize_ffn and any(x in name.lower() for x in ['ffn', 'mlp', 'feed']):
            return False

        # Only quantize Linear layers (weights and biases)
        return isinstance(module, nn.Linear)

    def _calculate_quantized_size(self, quantized: QuantizedTensor) -> int:
        """Calculate the memory size of a quantized tensor"""
        data_size = quantized.data.numel() * quantized.data.element_size()
        scales_size = quantized.scales.numel() * quantized.scales.element_size()
        zeros_size = quantized.zeros.numel() * quantized.zeros.element_size() if quantized.zeros is not None else 0

        return data_size + scales_size + zeros_size

    def validate_accuracy(self, original_model: nn.Module,
                         quantized_layers: Dict[str, QuantizedTensor],
                         test_inputs: List[torch.Tensor]) -> Dict[str, float]:
        """
        Validate quantized model accuracy against original

        Args:
            original_model: Original full-precision model
            quantized_layers: Quantized model layers
            test_inputs: Test input tensors

        Returns:
            Dictionary with accuracy metrics
        """
        print("Validating quantization accuracy...")

        original_model.eval()
        metrics = {}

        total_original_output = []
        total_quantized_output = []

        with torch.no_grad():
            for i, test_input in enumerate(test_inputs):
                print(f"  Processing test sample {i+1}/{len(test_inputs)}")

                # Get original model output
                original_output = original_model(test_input)
                total_original_output.append(original_output)

                # Create quantized model for inference
                quantized_model = self._create_quantized_model(original_model, quantized_layers)
                quantized_output = quantized_model(test_input)
                total_quantized_output.append(quantized_output)

        # Calculate metrics
        original_outputs = torch.cat(total_original_output, dim=0)
        quantized_outputs = torch.cat(total_quantized_output, dim=0)

        # MSE loss
        mse_loss = nn.functional.mse_loss(quantized_outputs, original_outputs).item()
        metrics['mse_loss'] = mse_loss

        # Cosine similarity (for logits)
        cos_sim = torch.nn.functional.cosine_similarity(
            quantized_outputs.flatten(),
            original_outputs.flatten(),
            dim=0
        ).item()
        metrics['cosine_similarity'] = cos_sim

        # Perplexity difference (if applicable)
        if quantized_outputs.shape[-1] > 1000:  # Likely logits
            original_probs = torch.softmax(original_outputs, dim=-1)
            quantized_probs = torch.softmax(quantized_outputs, dim=-1)

            # Simple perplexity approximation
            original_perplexity = torch.exp(-torch.mean(torch.log(torch.clamp(original_probs, min=1e-10))))
            quantized_perplexity = torch.exp(-torch.mean(torch.log(torch.clamp(quantized_probs, min=1e-10))))

            metrics['original_perplexity'] = original_perplexity.item()
            metrics['quantized_perplexity'] = quantized_perplexity.item()
            metrics['perplexity_ratio'] = quantized_perplexity.item() / original_perplexity.item()

        print("  Accuracy validation complete:")
        print(f"    MSE Loss: {mse_loss:.6f}")
        print(f"    Cosine Similarity: {cos_sim:.6f}")
        if 'perplexity_ratio' in metrics:
            print(f"    Perplexity Ratio: {metrics['perplexity_ratio']:.4f}")

        return metrics

    def _create_quantized_model(self, original_model: nn.Module,
                               quantized_layers: Dict[str, QuantizedTensor]) -> nn.Module:
        """Create a model that uses quantized weights during inference"""
        # This is a simplified implementation
        # In practice, you'd want to create a custom module that dequantizes on-the-fly

        quantized_model = type(original_model)(**original_model.__dict__)

        for name, module in quantized_model.named_modules():
            for param_name, _ in module.named_parameters():
                layer_name = f"{name}.{param_name}"
                if layer_name in quantized_layers:
                    quantized_param = self.block_quantizer.dequantize_tensor(quantized_layers[layer_name])
                    setattr(module, param_name, nn.Parameter(quantized_param))

        return quantized_model

    def save_quantized_model(self, quantized_layers: Dict[str, QuantizedTensor],
                           output_path: str, metadata: Optional[Dict] = None):
        """Save quantized model to disk with metadata"""

        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save quantized tensors
        for layer_name, quantized in quantized_layers.items():
            layer_path = output_dir / f"{layer_name.replace('.', '_')}.pt"
            torch.save({
                'data': quantized.data,
                'scales': quantized.scales,
                'zeros': quantized.zeros,
                'bits': quantized.bits,
                'block_size': quantized.block_size,
                'original_shape': quantized.original_shape
            }, layer_path)

        # Save metadata
        metadata_path = output_dir / "quantization_metadata.json"
        metadata = metadata or {}
        metadata.update({
            'quantization_config': {
                'bits': self.config.bits,
                'block_size': self.config.block_size,
                'symmetric': self.config.symmetric,
                'per_channel': self.config.per_channel,
                'quantize_attention': self.config.quantize_attention,
                'quantize_ffn': self.config.quantize_ffn,
                'quantize_embeddings': self.config.quantize_embeddings
            },
            'quantized_layers': list(quantized_layers.keys()),
            'total_layers': len(quantized_layers)
        })

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Quantized model saved to {output_dir}")
        print(f"  Layers quantized: {len(quantized_layers)}")

    def load_quantized_model(self, model_path: str) -> Tuple[Dict[str, QuantizedTensor], Dict]:
        """Load quantized model from disk"""

        model_dir = Path(model_path)

        # Load metadata
        metadata_path = model_dir / "quantization_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Load quantized tensors
        quantized_layers = {}
        for layer_name in metadata['quantized_layers']:
            layer_path = model_dir / f"{layer_name.replace('.', '_')}.pt"
            state_dict = torch.load(layer_path, map_location='cpu')

            quantized_layers[layer_name] = QuantizedTensor(
                data=state_dict['data'],
                scales=state_dict['scales'],
                zeros=state_dict.get('zeros'),
                bits=state_dict['bits'],
                block_size=state_dict['block_size'],
                original_shape=tuple(state_dict['original_shape'])
            )

        return quantized_layers, metadata

def create_lfm_quantization_config(target_memory_mb: int = 300) -> QuantizationConfig:
    """
    Create quantization config optimized for LFM models and memory constraints

    Args:
        target_memory_mb: Target memory usage in MB

    Returns:
        Optimized quantization configuration
    """
    if target_memory_mb <= 200:
        # Aggressive quantization for very constrained memory
        return QuantizationConfig(
            bits=4,
            block_size=32,
            symmetric=True,
            quantize_attention=True,
            quantize_ffn=True,
            quantize_embeddings=False
        )
    elif target_memory_mb <= 300:
        # Balanced quantization
        return QuantizationConfig(
            bits=4,
            block_size=64,
            symmetric=True,
            quantize_attention=True,
            quantize_ffn=True,
            quantize_embeddings=False
        )
    else:
        # Conservative quantization
        return QuantizationConfig(
            bits=8,
            block_size=128,
            symmetric=True,
            quantize_attention=True,
            quantize_ffn=True,
            quantize_embeddings=False
        )

# Example usage
if __name__ == "__main__":
    # Create test model
    model = nn.Sequential(
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1000)
    )

    # Create quantizer
    config = create_lfm_quantization_config(target_memory_mb=200)
    quantizer = LFMQuantizer(config)

    # Quantize model
    print("Quantizing model...")
    quantized_layers = quantizer.quantize_model(model)

    # Test basic functionality
    test_input = torch.randn(1, 1024)

    # Original inference
    with torch.no_grad():
        original_output = model(test_input)

    # Quantized inference
    quantized_model = quantizer._create_quantized_model(model, quantized_layers)
    with torch.no_grad():
        quantized_output = quantized_model(test_input)

    # Compare outputs
    mse = torch.mean((original_output - quantized_output) ** 2).item()
    cos_sim = torch.cosine_similarity(original_output.flatten(), quantized_output.flatten(), dim=0).item()

    print("\nQuantization Results:")
    print(f"  MSE Loss: {mse:.6f}")
    print(f"  Cosine Similarity: {cos_sim:.6f}")
    print(f"  Quantized layers: {len(quantized_layers)}")