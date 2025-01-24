from pathlib import Path
import torch
import psutil
import numpy as np
import time
import logging
from dataclasses import dataclass
from typing import Dict, Optional, List, Union, Tuple, Any

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    PretrainedConfig,
    Cache,
    DynamicCache, 
    OffloadedCache,
    QuantizedCache,
    QuantizedCacheConfig
)

class SimpleQuantizedCache(QuantizedCache):
    """Basic implementation of quantized cache using simple min-max quantization"""
    
    def _quantize(self, tensor: torch.Tensor, axis: int) -> torch.Tensor:
        # Simple min-max quantization
        with torch.no_grad():
            # Compute min and max along the specified axis
            if axis == 0:
                min_val = tensor.min(dim=0)[0]
                max_val = tensor.max(dim=0)[0]
            else:  # axis = -1
                min_val = tensor.min(dim=-1, keepdim=True)[0]
                max_val = tensor.max(dim=-1, keepdim=True)[0]
            
            # Scale to [0, 2^nbits - 1]
            scale = (max_val - min_val) / (2**self.nbits - 1)
            scale = torch.clamp(scale, min=1e-6)  # Prevent division by zero
            
            # Quantize
            qtensor = ((tensor - min_val) / scale).round().clamp(0, 2**self.nbits - 1)
            
            # Store scaling factors as attributes of the tensor
            qtensor.scale = scale
            qtensor.zero_point = min_val
            
            return qtensor
    
    def _dequantize(self, qtensor: torch.Tensor) -> torch.Tensor:
        # Dequantize using stored scale and zero point
        with torch.no_grad():
            return qtensor * qtensor.scale + qtensor.zero_point