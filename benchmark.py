import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import json
import psutil
import numpy as np
from dataclasses import dataclass, asdict, field
import logging
from pathlib import Path
from typing import List, Dict, Optional
import gc
from contextlib import contextmanager
import os
import statistics
import threading

os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class HardwareConfig:
    gpu_mem_total: int = 46068  # A40 memory in MB
    cpu_count: int = 8
    cpu_mem_total: int = 1024 * 1024  # 1TB in MB

@dataclass
class BenchmarkConfig:
    model_name: str = "facebook/opt-1.3b"
    context_lengths: List[int] = field(default_factory=lambda: [1024, 2048, 4096, 8192, 16384])
    output_lengths: List[int] = field(default_factory=lambda: [50, 100, 200])
    batch_size: int = 1
    num_runs: int = 5
    warmup_runs: int = 2
    output_dir: str = "benchmark_results"
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    decode_strategy: str = "gpu"  # Options: "gpu", "cpu"
    document_path: str = "data/crimeandpunishment.txt"

class ResourceMonitor:
    def __init__(self, sampling_rate: float = 0.1):
        self.sampling_rate = sampling_rate
        self.cpu_percentages = []
        self._stop_monitoring = False
    
    @staticmethod
    def get_gpu_memory_usage():
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1024**2,
                'reserved': torch.cuda.memory_reserved() / 1024**2,
                'max_allocated': torch.cuda.max_memory_allocated() / 1024**2
            }
        return {}

    @staticmethod
    def get_cpu_memory_usage():
        vm = psutil.virtual_memory()
        return {
            'total': vm.total / 1024**2,
            'available': vm.available / 1024**2,
            'used': vm.used / 1024**2,
            'cached': getattr(vm, 'cached', 0) / 1024**2
        }
    
    def _monitor_cpu(self):
        while not self._stop_monitoring:
            self.cpu_percentages.append(psutil.cpu_percent(percpu=True))
            time.sleep(self.sampling_rate)
    
    @contextmanager
    def track_resources(self):
        """Enhanced resource tracking with CPU utilization"""
        torch.cuda.reset_peak_memory_stats()
        self.cpu_percentages = []
        self._stop_monitoring = False
        resources = {}
        
        monitor_thread = threading.Thread(target=self._monitor_cpu)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        try:
            yield resources
        finally:
            self._stop_monitoring = True
            monitor_thread.join(timeout=1.0)
            
            gpu_mem = self.get_gpu_memory_usage()
            cpu_mem = self.get_cpu_memory_usage()
            
            if self.cpu_percentages:
                cpu_stats = {
                    'mean_per_core': [statistics.mean(core_vals) for core_vals in zip(*self.cpu_percentages)],
                    'max_per_core': [max(core_vals) for core_vals in zip(*self.cpu_percentages)],
                    'overall_mean': statistics.mean([sum(vals)/len(vals) for vals in self.cpu_percentages])
                }
            else:
                cpu_stats = {
                    'mean_per_core': [0],
                    'max_per_core': [0],
                    'overall_mean': 0
                }
            
            resources.update({
                'gpu': gpu_mem,
                'cpu': cpu_mem,
                'cpu_utilization': cpu_stats
            })

class ModelBenchmark:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.logger = self._setup_logging()
        self._setup_directories()
        self.monitor = ResourceMonitor()
        self.kv_cache = None        
        self.current_sequence_length = 0
        self.initial_sequence_length = 0
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logger

    def _setup_directories(self):
        """Create necessary directories"""
        Path(self.config.output_dir).mkdir(exist_ok=True)
        
    def load_model(self):
        """Load model to GPU initially"""
        self.logger.info(f"Loading model: {self.config.model_name}")
        
        try:
            # Always load to GPU first
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"  # Start with everything on GPU
            )
            
            # Debug: Log model architecture
            self.logger.info("Model architecture:")
            for name, module in self.model.named_children():
                self.logger.info(f"- {name}: {type(module)}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
            )
            
            model_size = sum(p.numel() for p in self.model.parameters()) * 2 / (1024**3)
            self.logger.info(f"Model size: {model_size:.2f} GB")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise

    def _log_state(self, phase: str):
        """Log current state for debugging"""
        self.logger.info(f"\n=== State during {phase} ===")
        if self.kv_cache is not None:
            # Log KV cache details
            kv_first_layer = self.kv_cache[0]
            k, v = kv_first_layer
            self.logger.info(f"KV cache first layer shapes: K={k.shape}, V={v.shape}")
            self.logger.info(f"KV cache device: {k.device}")
        self.logger.info(f"Current sequence length: {self.current_sequence_length}")
        
    def prefill_phase(self, input_ids):
        """Run prefill phase on GPU"""
        self.logger.info("Running prefill phase on GPU")
        
        with torch.no_grad():
            # Store initial sequence length
            self.initial_sequence_length = input_ids.shape[1]
            self.current_sequence_length = self.initial_sequence_length
            
            self._log_state("before prefill")
            
            outputs = self.model(
                input_ids,
                use_cache=True,
                return_dict=True
            )
            
            self.kv_cache = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            
            self._log_state("after prefill")
            
        return next_token_logits

    def prepare_decode_strategy(self):
        """Prepare model for decode phase based on strategy"""
        if self.config.decode_strategy == "cpu":
            self.logger.info("Moving decoder to CPU for decode phase")
            
            # For OPT models, the decoder is at model.model.decoder
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'decoder'):
                self.model.model.decoder.to("cpu")
                
                if self.kv_cache is not None:
                    # Move KV cache to CPU and log shapes before/after
                    self._log_state("before moving KV cache")
                    self.kv_cache = tuple(
                        tuple(t.to("cpu") for t in layer)
                        for layer in self.kv_cache
                    )
                    self._log_state("after moving KV cache")
            else:
                self.logger.warning("Could not find decoder in expected location")

    def decode_step(self, input_ids, attention_mask=None):
        """Single decode step using stored KV cache"""
        with torch.no_grad():
            device = "cpu" if self.config.decode_strategy == "cpu" else "cuda"
            
            # Ensure input_ids has the right shape
            if len(input_ids.shape) == 1:
                input_ids = input_ids.unsqueeze(0)
            if len(input_ids.shape) == 2 and input_ids.shape[1] != 1:
                input_ids = input_ids[:, -1:]
                
            input_ids = input_ids.to(device)
            
            # Get sequence length from KV cache
            k, v = self.kv_cache[0]  # First layer
            past_seq_len = k.shape[2]  # seq_len dimension in KV cache
            
            # Create attention mask including both past and current token
            batch_size = input_ids.shape[0]
            total_seq_len = past_seq_len + 1  # Include current token
            
            full_attention_mask = torch.ones(
                (batch_size, total_seq_len),
                dtype=torch.long,
                device=device
            )
            
            self.logger.info(f"Attention mask shape: {full_attention_mask.shape}")
            self.logger.info(f"Past sequence length: {past_seq_len}")
            self.logger.info(f"Total sequence length: {total_seq_len}")
            
            self._log_state("before decode step")
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=full_attention_mask,
                use_cache=True,
                past_key_values=self.kv_cache,
                return_dict=True
            )
            
            # Update KV cache
            self.kv_cache = outputs.past_key_values
            self.current_sequence_length = total_seq_len
            
            self._log_state("after decode step")
            
            return outputs.logits[:, -1, :]

    def run_single_generation(self, tokens, output_length: int):
        """Run full generation with separate prefill and decode phases"""
        with self.monitor.track_resources() as resources:
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                # Reset sequence tracking
                self.current_sequence_length = 0
                
                # Prefill phase
                input_ids = tokens.input_ids
                next_token_logits = self.prefill_phase(input_ids)
                
                # Prepare decode strategy
                self.prepare_decode_strategy()
                
                # Generate tokens
                generated_tokens = []
                for i in range(output_length):
                    self.logger.info(f"Generating token {i+1}/{output_length}")
                    next_token = torch.argmax(next_token_logits, dim=-1)
                    generated_tokens.append(next_token)
                    
                    # Prepare input for next decode step
                    current_input = next_token.unsqueeze(0)
                    
                    # Decode step
                    next_token_logits = self.decode_step(current_input)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            # Move back to GPU for next run if needed
            if self.config.decode_strategy == "cpu":
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'decoder'):
                    self.model.model.decoder.to("cuda")
        
        return {
            'time': end_time - start_time,
            'resources': resources,
            'output_length': len(generated_tokens)
        }
        

    def clean_memory(self):
        """Clean up GPU memory between runs"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()

    def run_benchmark(self):
        """Run the complete benchmark suite"""
        self.load_model()
        results = []
        
        for context_length in self.config.context_lengths:
            for output_length in self.config.output_lengths:
                self.logger.info(f"\nTesting context length: {context_length}, output length: {output_length}")
                self.clean_memory()
                
                # Prepare input
                tokens = self.tokenizer(
                    sample_text,
                    truncation=True,
                    max_length=context_length,
                    return_tensors="pt"
                ).to("cuda")
                
                # Warmup
                self.logger.info("Performing warmup runs...")
                for _ in range(self.config.warmup_runs):
                    _ = self.run_single_generation(tokens, output_length)
                
                # Benchmark runs
                run_results = []
                for run in range(self.config.num_runs):
                    self.logger.info(f"Run {run + 1}/{self.config.num_runs}")
                    try:
                        result = self.run_single_generation(tokens, output_length)
                        run_results.append(result)
                        
                        self.logger.info(f"Generation time: {result['time']:.2f}s")
                        self.logger.info(f"Tokens per second: {output_length/result['time']:.2f}")
                        
                        cpu_util = result['resources']['cpu_utilization']['overall_mean']
                        self.logger.info(f"CPU utilization: {cpu_util:.1f}%")
                    except Exception as e:
                        self.logger.error(f"Error in run {run + 1}: {str(e)}")
                        continue
                
                if not run_results:
                    continue
                    
                # Aggregate results
                result = {
                    'context_length': context_length,
                    'output_length': output_length,
                    'avg_time': statistics.mean([r['time'] for r in run_results]),
                    'std_time': statistics.stdev([r['time'] for r in run_results]) if len(run_results) > 1 else 0,
                    'tokens_per_second': output_length / statistics.mean([r['time'] for r in run_results]),
                    'gpu_memory_peak': max([r['resources']['gpu']['max_allocated'] for r in run_results]),
                    'cpu_utilization': {
                        'mean': statistics.mean([r['resources']['cpu_utilization']['overall_mean'] for r in run_results]),
                        'peak': max([max(r['resources']['cpu_utilization']['max_per_core']) for r in run_results])
                    }
                }
                
                results.append(result)
        
        return results