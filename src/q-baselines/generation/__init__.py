"""
Quantized Baseline Generation Package
Implements Q-Diffusion and PTQ4DM quantization for SDXL
"""

from .qdiffusion import QDiffusionQuantizer, create_qdiffusion_quantizer
from .ptq4dm import PTQ4DMQuantizer, create_ptq4dm_quantizer
from .quant_utils import (
    QuantizedLinear,
    QuantizedConv2d,
    TimestepAwareQuantizedLayer,
    replace_layers_with_quantized,
    disable_quantization,
    enable_quantization,
    print_quantization_stats,
)

__all__ = [
    "QDiffusionQuantizer",
    "create_qdiffusion_quantizer",
    "PTQ4DMQuantizer",
    "create_ptq4dm_quantizer",
    "QuantizedLinear",
    "QuantizedConv2d",
    "TimestepAwareQuantizedLayer",
    "replace_layers_with_quantized",
]
