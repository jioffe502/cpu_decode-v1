from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from caching.cache_config import CacheConfig
from benchmarks.metrics_tracker import EnhancedMetricsTracker

class LlamaInference:
    def __init__(
        self, 
        model_name: str = "meta-llama/Llama-3.2-1B",
        cache_config: Optional[CacheConfig] = None,
    ):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache_config = cache_config or CacheConfig(strategy="dynamic")
        self.metrics_tracker = None
        self.setup()
    
    def setup(self):
        """Initialize model and tokenizer with specific decode strategy"""
        logger.info(f"Loading model '{self.model_name}' with cache strategy '{self.cache_config.strategy}'")
        
        # Clear GPU memory before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto" if not self.cache_config.decode_on_cpu else None,
            low_cpu_mem_usage=True
        )
        
        # Move decoder to CPU if specified
        if self.cache_config.decode_on_cpu:
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'decoder'):
                self.model.model.decoder.to("cpu")
                logger.info("Moved decoder to CPU")
            else:
                logger.warning("Could not find decoder module for CPU offloading")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Initialize cache
        self.cache = CacheConfig.get_cache(self.cache_config, self.model.config)
        
    def _get_cache_metrics(self) -> Dict[str, float]:
        """Collect cache-specific metrics based on cache type"""
        metrics = {
            'cache_size': self.cache.get_seq_length() if self.cache else 0,
        }
        
        if isinstance(self.cache, SimpleQuantizedCache):
            # Add quantization-specific metrics
            metrics.update({
                'residual_length': self.cache.residual_length,
                'nbits': self.cache.nbits,
            })
        elif isinstance(self.cache, OffloadedCache):
            # Add offloading-specific metrics
            metrics['is_on_gpu'] = self.cache.key_cache[0].device.type == 'cuda' if self.cache.key_cache else False
        
        return metrics
        
    def _get_detailed_cache_metrics(self) -> Dict[str, float]:
        """Get detailed cache-specific metrics"""
        metrics = {}
        
        if isinstance(self.cache, SimpleQuantizedCache):
            # Get quantization-specific metrics
            metrics.update({
                'quantization_bits': self.cache.nbits,
                'residual_cache_size': self.cache.residual_length,
                'total_cache_tokens': len(self.cache.key_cache) * self.cache.get_seq_length(),
                'memory_savings': 1.0 - (self.cache.nbits / 32.0)  # Compared to FP32
            })
            
        elif isinstance(self.cache, DynamicCache):
            # Get dynamic cache metrics
            total_cached = sum(k.nelement() * k.element_size() for k in self.cache.key_cache) / 1024**2
            metrics.update({
                'total_cache_size_mb': total_cached,
                'tokens_cached': self.cache.get_seq_length(),
                'memory_per_token': total_cached / max(1, self.cache.get_seq_length())
            })
            
        return metrics

    def _format_time(self, ns):
        """Helper to format nanoseconds into readable units"""
        if ns < 1000:
            return f"{ns:.2f}ns"
        elif ns < 1000000:
            return f"{ns/1000:.2f}µs"
        else:
            return f"{ns/1000000:.2f}ms"

    def run_inference(
        self,
        input_text: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
    ) -> Dict:
        """Run inference with enhanced metrics tracking"""
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt",
            truncation=True
        ).to(self.device)
        
        self.metrics_tracker = EnhancedMetricsTracker()
        decode_timings = []  # Store detailed decode timings
        
        try:
            with torch.inference_mode():
                # Prefill phase
                self.metrics_tracker.start_phase('prefill')
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    
                prefill_start = time.perf_counter_ns()
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=self.cache,
                    use_cache=True
                )
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                prefill_latency = (time.perf_counter_ns() - prefill_start) / 1e9
                
                # Record prefill metrics
                cache_metrics = self._get_cache_metrics()
                self.metrics_tracker.end_phase(
                    tokens_processed=input_ids.shape[1],
                    cache_metrics=cache_metrics
                )
                
                # Enhanced decode phase metrics
                self.metrics_tracker.start_phase('decode')
                
                for i in range(max_new_tokens):
                    decode_metrics = {}
                    
                    # Detailed timing for decode steps
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    token_start = time.perf_counter_ns()
                    
                    # 1. Get next token logits
                    next_token_logits = outputs.logits[:, -1, :] / temperature
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    logits_time = time.perf_counter_ns()
                    decode_metrics['logits_latency'] = (logits_time - token_start) / 1e9
                    
                    # 2. Sample next token
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    sampling_time = time.perf_counter_ns()
                    decode_metrics['sampling_latency'] = (sampling_time - logits_time) / 1e9
                    
                    # 3. Prepare next inputs
                    input_ids = next_token
                    attention_mask = torch.cat([attention_mask, attention_mask.new_ones((1, 1))], dim=-1)
                    
                    # 4. Forward pass
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    forward_start = time.perf_counter_ns()
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=self.cache,
                        use_cache=True
                    )
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    forward_end = time.perf_counter_ns()
                    decode_metrics['forward_latency'] = (forward_end - forward_start) / 1e9
                    
                    # Total token generation time
                    total_latency = (forward_end - token_start) / 1e9
                    
                    # Get cache metrics
                    cache_metrics = self._get_cache_metrics()
                    cache_metrics.update({
                        'device': self.cache.key_cache[0].device.type if hasattr(self.cache, 'key_cache') else 'unknown',
                        'decode_step_breakdown': decode_metrics
                    })
                    
                    # Record detailed token metrics
                    self.metrics_tracker.sample_token(
                        token_index=i,
                        phase='decode',
                        latency=total_latency,
                        cache_size=cache_metrics['cache_size'],
                        cache_metrics=cache_metrics
                    )
                    
                    decode_timings.append(decode_metrics)
                    
                    # Check for EOS
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                
                # End decode phase with detailed metrics
                final_cache_metrics = self._get_cache_metrics()
                final_cache_metrics['decode_timing_stats'] = {
                    'mean_logits_latency': np.mean([t['logits_latency'] for t in decode_timings]),
                    'mean_sampling_latency': np.mean([t['sampling_latency'] for t in decode_timings]),
                    'mean_forward_latency': np.mean([t['forward_latency'] for t in decode_timings]),
                    'p90_forward_latency': np.percentile([t['forward_latency'] for t in decode_timings], 90),
                    'p99_forward_latency': np.percentile([t['forward_latency'] for t in decode_timings], 99),
                }
                
                self.metrics_tracker.end_phase(
                    tokens_processed=i + 1,
                    cache_metrics=final_cache_metrics
                )
                
                # Get generated text
                generated_text = self.tokenizer.decode(
                    input_ids[0], 
                    skip_special_tokens=True
                )
                
                return {
                    "text": generated_text,
                    "metrics": self.metrics_tracker.get_summary(),
                    "decode_timings": decode_timings
                }
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def cleanup(self):
        """Clean up resources"""
        del self.model
        del self.cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()