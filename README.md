# LLM Hybrid CPU-GPU Inference

This repository contains research code exploring hybrid CPU-GPU inference strategies for LLMs, focusing on enabling ultra-long context windows through CPU decode offloading and optimized KV cache management.

The project investigates three key approaches:
- Dynamic KV cache with GPU-only execution
- Offloaded KV cache with CPU-GPU hybrid execution 
- Quantized KV cache for memory optimization

## Key Results

- Achieved 32K+ token context lengths on consumer GPUs
- CPU decode maintains ~30% of GPU throughput at large contexts
- Successful processing of 131K token contexts with CPU offloading

## Requirements

```
torch>=2.0.0
transformers>=4.36.0
numpy>=1.24.0
pandas>=2.0.0
psutil>=5.9.0
```

## Contact

Jacob Ioffe - ji97@cornell.edu
