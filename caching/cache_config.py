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

class CacheConfig:
    def __init__(self, strategy, decode_on_cpu=False, quantization=None):
        self.strategy = strategy
        self.decode_on_cpu = decode_on_cpu
        self.quantization = quantization

    @staticmethod
    def get_cache(config, model_config):
        if config.strategy == "dynamic":
            return DynamicCache()
        elif config.strategy == "quantized":
            quant_config = QuantizedCacheConfig(**config.quantization)
            return QuantizedCache(cache_config=quant_config)
        return None