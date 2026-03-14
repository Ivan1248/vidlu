"""
VLM Inference Speed Benchmark for Qwen3-VL models.

This module provides a comprehensive benchmark for measuring the effect of:
- Image size (visual tokens)
- Input text length (text tokens)
- Output length (generated tokens)

The benchmark uses a factor-isolation design to compute per-token cost coefficients
using linear regression.

Usage:
    # Run quick benchmark (~20 min)
    python -m vidlu_irap_gaim.tools.vlm_benchmark --preset quick

    # Run full benchmark (~1.5 hours)
    python -m vidlu_irap_gaim.tools.vlm_benchmark --preset full

    # Analyze existing results only
    python -m vidlu_irap_gaim.tools.vlm_benchmark --analyze-only results.json

    # Generate plots
    python -m vidlu_irap_gaim.tools.vlm_benchmark --analyze-only results.json --plot
"""
import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# =============================================================================
# Data Classes for Benchmark Results
# =============================================================================

@dataclass
class BenchmarkRun:
    """Single benchmark measurement with actual token counts."""
    # Configuration
    image_size: tuple[int, int]
    num_attributes: int
    max_new_tokens: int
    batch_size: int
    sweep_type: str  # "visual", "text", or "output"
    run_index: int
    
    # Measured token counts
    visual_tokens: int
    text_tokens: int
    prompt_tokens: int  # visual + text (total input)
    output_tokens: int  # actual generated
    
    # Timing measurements (milliseconds)
    ttft_ms: float  # time to first token
    generation_ms: float  # decode phase only
    total_ms: float  # end-to-end
    
    # Memory
    peak_memory_gb: float
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class FactorEffect:
    """Regression result for one factor."""
    factor_name: str
    coefficient: float  # ms per token
    std_error: float
    r_squared: float
    intercept: float
    data_points: int


@dataclass
class BenchmarkSummary:
    """Summary of benchmark results with factor analysis."""
    model_id: str
    gpu_name: str
    backend: str
    
    # Factor effects
    visual_token_effect: FactorEffect | None
    text_token_effect: FactorEffect | None
    output_token_effect: FactorEffect | None
    
    # Raw data
    num_runs: int
    total_time_s: float
    
    # Prediction model coefficients
    intercept: float
    beta_visual: float
    beta_text: float
    beta_output: float


# =============================================================================
# Benchmark Configuration
# =============================================================================

@dataclass
class SweepConfig:
    """Configuration for a single sweep."""
    name: str
    image_sizes: list[tuple[int, int]]
    num_attributes_list: list[int]
    max_new_tokens_list: list[int]
    num_runs: int
    batch_size: int = 1


def get_quick_preset() -> dict[str, SweepConfig]:
    """Quick benchmark preset (~20 min)."""
    return {
        "visual": SweepConfig(
            name="visual",
            # Wider range: 256 to 4096 visual tokens
            image_sizes=[(224, 224), (448, 448), (672, 672), (896, 896)],
            num_attributes_list=[10],  # Fixed
            max_new_tokens_list=[256],  # Fixed
            num_runs=3,
        ),
        "text": SweepConfig(
            name="text",
            image_sizes=[(336, 336)],  # Fixed
            num_attributes_list=[3, 10, 20, 41],
            max_new_tokens_list=[256],  # Fixed
            num_runs=3,
        ),
        "output": SweepConfig(
            name="output",
            image_sizes=[(336, 336)],  # Fixed
            num_attributes_list=[6],  # Fixed
            max_new_tokens_list=[64, 128, 256, 512],
            num_runs=3,
        ),
    }


def get_full_preset() -> dict[str, SweepConfig]:
    """Full benchmark preset (~1.5 hours)."""
    return {
        "visual": SweepConfig(
            name="visual",
            # Wide range: 256 to 5476 visual tokens
            image_sizes=[
                (224, 224), (336, 336), (448, 448),
                (672, 672), (896, 896), (1024, 1024)
            ],
            num_attributes_list=[10],  # Fixed
            max_new_tokens_list=[256],  # Fixed
            num_runs=5,
        ),
        "text": SweepConfig(
            name="text",
            image_sizes=[(336, 336)],  # Fixed
            num_attributes_list=[3, 6, 12, 24, 41],
            max_new_tokens_list=[256],  # Fixed
            num_runs=5,
        ),
        "output": SweepConfig(
            name="output",
            image_sizes=[(336, 336)],  # Fixed
            num_attributes_list=[6],  # Fixed
            max_new_tokens_list=[64, 128, 256, 512, 1024],
            num_runs=5,
        ),
    }


# =============================================================================
# Token Counting Utilities
# =============================================================================

def estimate_visual_tokens(image_size: tuple[int, int], patch_size: int = 14) -> int:
    """Estimate visual tokens for an image size.
    
    Qwen-VL uses patch_size=14 by default.
    Visual tokens = ceil(height/patch_size) * ceil(width/patch_size)
    """
    width, height = image_size
    h_patches = int(np.ceil(height / patch_size))
    w_patches = int(np.ceil(width / patch_size))
    return h_patches * w_patches


def create_test_image(size: tuple[int, int], seed: int = 42) -> Image.Image:
    """Create a random test image of the specified size."""
    np.random.seed(seed)
    width, height = size
    # Create a simple gradient + noise image
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)
    
    # RGB channels with different patterns
    r = (xx * 255).astype(np.uint8)
    g = (yy * 255).astype(np.uint8)
    b = ((xx + yy) / 2 * 255).astype(np.uint8)
    
    # Add some noise
    noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
    img_array = np.stack([r, g, b], axis=-1) + noise
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_array)


# =============================================================================
# Benchmark Runner
# =============================================================================

class VLMBenchmark:
    """VLM inference speed benchmark."""
    
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
        backend: str = "vllm",
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 8192,
        output_dir: str = "benchmark_results",
        debug: bool = False,
    ):
        self.model_id = model_id
        self.backend = backend
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.output_dir = Path(output_dir)
        self.debug = debug
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Lazy-loaded components
        self._predictor = None
        self._processor = None
        self._attr_to_value_to_class_idx = None
        self._all_attributes = None
        
    def _load_predictor(self):
        """Load the VLM predictor."""
        if self._predictor is not None:
            return
            
        print(f"[Benchmark] Loading model: {self.model_id} (backend={self.backend})")
        
        if self.backend == "vllm":
            from vidlu_irap_gaim.vlm import Qwen3VLvLLMPredictor
            self._predictor = Qwen3VLvLLMPredictor(
                model_id=self.model_id,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                max_new_tokens=1024,  # Will be overridden per run
                chunk_size=100,  # Large to avoid chunking in benchmarks
                debug=self.debug,
            )
        else:
            from vidlu_irap_gaim.vlm import Qwen3VLPredictor
            self._predictor = Qwen3VLPredictor(
                model_id=self.model_id,
                device="cuda",
                torch_dtype="bfloat16",
                use_flash_attention=True,
                max_new_tokens=1024,
                chunk_size=100,
                debug=self.debug,
            )
        
        # Force model load
        self._predictor._load_model()
        print("[Benchmark] Model loaded successfully")
        
    def _load_attribute_metadata(self):
        """Load attribute metadata for prompt generation."""
        if self._attr_to_value_to_class_idx is not None:
            return
            
        from vidlu_irap_gaim.datasets import make_bih_data
        
        print("[Benchmark] Loading attribute metadata...")
        data = make_bih_data()
        ref_ds = data["test"]
        self._attr_to_value_to_class_idx = ref_ds.info.attr_to_value_to_class_idx
        self._all_attributes = list(self._attr_to_value_to_class_idx.keys())
        print(f"[Benchmark] Loaded {len(self._all_attributes)} attributes")
        
    def _get_gpu_info(self) -> str:
        """Get GPU name."""
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "Unknown GPU"
    
    def _get_peak_memory_gb(self) -> float:
        """Get peak GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
        return 0.0
    
    def _reset_memory_stats(self):
        """Reset GPU memory stats."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    
    def _count_tokens(
        self,
        image: Image.Image,
        prompt: str,
    ) -> tuple[int, int, int]:
        """Count visual tokens, text tokens, and total prompt tokens.
        
        Returns:
            (visual_tokens, text_tokens, prompt_tokens)
        """
        # Visual tokens estimated from image size
        visual_tokens = estimate_visual_tokens(image.size)
        
        # Text tokens from processor/tokenizer
        if hasattr(self._predictor, '_processor') and self._predictor._processor is not None:
            processor = self._predictor._processor
            # Tokenize just the text prompt
            text_encoding = processor.tokenizer(prompt, return_tensors="pt")
            text_tokens = text_encoding.input_ids.shape[1]
        else:
            # Fallback estimation: ~4 chars per token
            text_tokens = len(prompt) // 4
        
        # Total prompt tokens
        prompt_tokens = visual_tokens + text_tokens
        
        return visual_tokens, text_tokens, prompt_tokens
    
    def _run_single_inference(
        self,
        image: Image.Image,
        attrs_to_include: list[str],
        max_new_tokens: int,
    ) -> tuple[str, int, float, float, float]:
        """Run a single inference and measure timing.
        
        Returns:
            (response, output_tokens, ttft_ms, generation_ms, total_ms)
        """
        from vidlu_irap_gaim.vlm import StandardResponseScheme
        from vidlu_irap_gaim.vlm.prompts import DEFAULT_DETAIL_LEVEL

        # Build prompt
        scheme = StandardResponseScheme(self._attr_to_value_to_class_idx)
        prompt = scheme.build_prompt(attrs_to_include, detail_level=DEFAULT_DETAIL_LEVEL)
        
        # Update max_new_tokens for this run
        original_max_tokens = self._predictor.max_new_tokens
        self._predictor.max_new_tokens = max_new_tokens
        if hasattr(self._predictor, '_sampling_params') and self._predictor._sampling_params is not None:
            self._predictor._sampling_params.max_tokens = max_new_tokens
        
        # Prepare input for vLLM
        if self.backend == "vllm":
            vllm_input = self._predictor._prepare_vllm_input(image, prompt)
            
            # Time the generation
            self._reset_memory_stats()
            
            start_time = time.perf_counter()
            outputs = self._predictor._llm.generate(
                [vllm_input], 
                self._predictor._sampling_params, 
                use_tqdm=False
            )
            end_time = time.perf_counter()
            
            total_ms = (end_time - start_time) * 1000
            
            # Extract response
            response = outputs[0].outputs[0].text
            output_tokens = len(outputs[0].outputs[0].token_ids)
            
            # vLLM doesn't provide TTFT directly in basic usage
            # Estimate: prefill time is proportional to input tokens
            # For now, we'll use a heuristic
            if output_tokens > 0:
                ttft_ms = total_ms * 0.3  # Rough estimate: 30% prefill
                generation_ms = total_ms * 0.7
            else:
                ttft_ms = total_ms
                generation_ms = 0
        else:
            # HuggingFace backend
            self._reset_memory_stats()
            
            start_time = time.perf_counter()
            response = self._predictor._generate_single(image, prompt)
            end_time = time.perf_counter()
            
            total_ms = (end_time - start_time) * 1000
            
            # Estimate output tokens (rough: 4 chars per token)
            output_tokens = len(response) // 4
            ttft_ms = total_ms * 0.3
            generation_ms = total_ms * 0.7
        
        # Restore original max_tokens
        self._predictor.max_new_tokens = original_max_tokens
        
        return response, output_tokens, ttft_ms, generation_ms, total_ms
    
    def run_sweep(
        self,
        config: SweepConfig,
        warmup_runs: int = 2,
    ) -> list[BenchmarkRun]:
        """Run a single sweep and collect measurements."""
        self._load_predictor()
        self._load_attribute_metadata()
        
        results = []
        sweep_type = config.name
        
        # Generate all configurations
        configs = []
        for image_size in config.image_sizes:
            for num_attrs in config.num_attributes_list:
                for max_tokens in config.max_new_tokens_list:
                    configs.append((image_size, num_attrs, max_tokens))
        
        print(f"\n[Sweep: {sweep_type}] Running {len(configs)} configurations x {config.num_runs} runs")
        print(f"  Image sizes: {config.image_sizes}")
        print(f"  Attributes: {config.num_attributes_list}")
        print(f"  Max tokens: {config.max_new_tokens_list}")
        
        # Warmup
        if warmup_runs > 0:
            print(f"\n[Warmup] Running {warmup_runs} warmup iterations...")
            warmup_image = create_test_image(config.image_sizes[0])
            warmup_attrs = self._all_attributes[:config.num_attributes_list[0]]
            for i in range(warmup_runs):
                self._run_single_inference(warmup_image, warmup_attrs, 64)
            print("[Warmup] Complete")
        
        # Main benchmark loop
        for cfg_idx, (image_size, num_attrs, max_tokens) in enumerate(configs):
            print(f"\n[Config {cfg_idx + 1}/{len(configs)}] "
                  f"Image={image_size}, Attrs={num_attrs}, MaxTokens={max_tokens}")
            
            # Create test image
            test_image = create_test_image(image_size)
            
            # Select attributes
            attrs_to_include = self._all_attributes[:num_attrs]
            
            # Build prompt for token counting
            from vidlu_irap_gaim.vlm import StandardResponseScheme
            from vidlu_irap_gaim.vlm.prompts import DEFAULT_DETAIL_LEVEL
            scheme = StandardResponseScheme(self._attr_to_value_to_class_idx)
            prompt = scheme.build_prompt(attrs_to_include, detail_level=DEFAULT_DETAIL_LEVEL)
            
            # Count tokens
            visual_tokens, text_tokens, prompt_tokens = self._count_tokens(
                test_image, prompt
            )
            
            for run_idx in range(config.num_runs):
                # Run inference
                response, output_tokens, ttft_ms, generation_ms, total_ms = \
                    self._run_single_inference(test_image, attrs_to_include, max_tokens)
                
                # Get memory
                peak_memory_gb = self._get_peak_memory_gb()
                
                # Create result
                result = BenchmarkRun(
                    image_size=image_size,
                    num_attributes=num_attrs,
                    max_new_tokens=max_tokens,
                    batch_size=config.batch_size,
                    sweep_type=sweep_type,
                    run_index=run_idx,
                    visual_tokens=visual_tokens,
                    text_tokens=text_tokens,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    ttft_ms=ttft_ms,
                    generation_ms=generation_ms,
                    total_ms=total_ms,
                    peak_memory_gb=peak_memory_gb,
                )
                results.append(result)
                
                if self.debug:
                    print(f"  Run {run_idx + 1}: {total_ms:.0f}ms, "
                          f"output={output_tokens} tokens")
            
            # Print average for this config
            config_runs = [r for r in results 
                          if r.image_size == image_size 
                          and r.num_attributes == num_attrs
                          and r.max_new_tokens == max_tokens]
            avg_ms = np.mean([r.total_ms for r in config_runs])
            avg_output = np.mean([r.output_tokens for r in config_runs])
            print(f"  Average: {avg_ms:.0f}ms, output={avg_output:.0f} tokens")
        
        return results


# =============================================================================
# Statistical Analysis
# =============================================================================

def linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Simple OLS linear regression.
    
    Returns:
        (slope, intercept, r_squared, std_error)
    """
    n = len(x)
    if n < 2:
        return 0.0, 0.0, 0.0, float('inf')
    
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calculate slope and intercept
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    if denominator == 0:
        return 0.0, y_mean, 0.0, float('inf')
    
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    # Calculate R-squared
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Calculate standard error of slope
    if n > 2:
        mse = ss_res / (n - 2)
        se_slope = np.sqrt(mse / denominator) if denominator > 0 else float('inf')
    else:
        se_slope = float('inf')
    
    return slope, intercept, r_squared, se_slope


def analyze_factor_effects(runs: list[BenchmarkRun]) -> dict[str, FactorEffect]:
    """Compute per-token cost for each factor using linear regression."""
    effects = {}
    
    # Group by sweep type
    visual_runs = [r for r in runs if r.sweep_type == "visual"]
    text_runs = [r for r in runs if r.sweep_type == "text"]
    output_runs = [r for r in runs if r.sweep_type == "output"]
    
    # Analyze visual token effect
    if visual_runs:
        x = np.array([r.visual_tokens for r in visual_runs])
        y = np.array([r.total_ms for r in visual_runs])
        slope, intercept, r_sq, se = linear_regression(x, y)
        effects["visual"] = FactorEffect(
            factor_name="visual_tokens",
            coefficient=slope,
            std_error=se,
            r_squared=r_sq,
            intercept=intercept,
            data_points=len(visual_runs),
        )
    
    # Analyze text token effect
    if text_runs:
        x = np.array([r.text_tokens for r in text_runs])
        y = np.array([r.total_ms for r in text_runs])
        slope, intercept, r_sq, se = linear_regression(x, y)
        effects["text"] = FactorEffect(
            factor_name="text_tokens",
            coefficient=slope,
            std_error=se,
            r_squared=r_sq,
            intercept=intercept,
            data_points=len(text_runs),
        )
    
    # Analyze output token effect
    if output_runs:
        x = np.array([r.output_tokens for r in output_runs])
        y = np.array([r.total_ms for r in output_runs])
        slope, intercept, r_sq, se = linear_regression(x, y)
        effects["output"] = FactorEffect(
            factor_name="output_tokens",
            coefficient=slope,
            std_error=se,
            r_squared=r_sq,
            intercept=intercept,
            data_points=len(output_runs),
        )
    
    return effects


def compute_combined_model(effects: dict[str, FactorEffect]) -> tuple[float, float, float, float]:
    """Compute combined prediction model coefficients.
    
    Returns:
        (intercept, beta_visual, beta_text, beta_output)
    """
    # Use average intercept from all effects
    intercepts = [e.intercept for e in effects.values() if e is not None]
    avg_intercept = np.mean(intercepts) if intercepts else 0.0
    
    beta_visual = effects.get("visual", FactorEffect("", 0, 0, 0, 0, 0)).coefficient
    beta_text = effects.get("text", FactorEffect("", 0, 0, 0, 0, 0)).coefficient
    beta_output = effects.get("output", FactorEffect("", 0, 0, 0, 0, 0)).coefficient
    
    return avg_intercept, beta_visual, beta_text, beta_output


def compute_text_output_model(
    runs: list[BenchmarkRun],
    effects: dict[str, FactorEffect] | None = None,
) -> tuple[float, float, float, float]:
    """Compute a text+output model using coefficients from individual sweeps.
    
    Rather than fitting multivariate regression on mismatched sweep data
    (which causes coefficient instability), we combine the coefficients
    from individual single-variable sweeps.
    
    Returns:
        (intercept, beta_text, beta_output, r_squared)
    """
    # If effects not provided, compute them
    if effects is None:
        effects = analyze_factor_effects(runs)
    
    text_effect = effects.get("text")
    output_effect = effects.get("output")
    
    if text_effect is None or output_effect is None:
        return 0.0, 0.0, 0.0, 0.0
    
    # Use coefficients from individual sweeps
    beta_text = text_effect.coefficient
    beta_output = output_effect.coefficient
    
    # Compute intercept: use average of individual intercepts, but adjust
    # for the baseline conditions of each sweep
    # Text sweep: image=336x336, output=256
    # Output sweep: image=336x336, text=~500
    # 
    # For a clean combined model, we want:
    #   latency = α + β_text × text + β_output × output
    # 
    # From text sweep (with fixed output ~256):
    #   latency = text_intercept + β_text × text
    #   So: text_intercept ≈ α + β_output × 256
    #   Thus: α ≈ text_intercept - β_output × 256
    #
    # From output sweep (with fixed text ~500):
    #   latency = output_intercept + β_output × output
    #   So: output_intercept ≈ α + β_text × 500
    #   Thus: α ≈ output_intercept - β_text × 500
    
    # Estimate baseline text and output tokens from the sweeps
    text_runs = [r for r in runs if r.sweep_type == "text"]
    output_runs = [r for r in runs if r.sweep_type == "output"]
    
    # Get typical output tokens in text sweep
    if text_runs:
        typical_output_in_text_sweep = np.mean([r.output_tokens for r in text_runs])
    else:
        typical_output_in_text_sweep = 256
    
    # Get typical text tokens in output sweep
    if output_runs:
        typical_text_in_output_sweep = np.mean([r.text_tokens for r in output_runs])
    else:
        typical_text_in_output_sweep = 500
    
    # Compute two estimates of α and average them
    alpha_from_text = text_effect.intercept - beta_output * typical_output_in_text_sweep
    alpha_from_output = output_effect.intercept - beta_text * typical_text_in_output_sweep
    intercept = (alpha_from_text + alpha_from_output) / 2
    
    # Compute R² by validating on the combined data
    all_runs = text_runs + output_runs
    if len(all_runs) > 0:
        y_true = np.array([r.total_ms for r in all_runs])
        y_pred = np.array([
            intercept + beta_text * r.text_tokens + beta_output * r.output_tokens
            for r in all_runs
        ])
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    else:
        r_squared = 0.0
    
    return intercept, beta_text, beta_output, r_squared


# =============================================================================
# Report Generation
# =============================================================================

def generate_summary_report(
    runs: list[BenchmarkRun],
    effects: dict[str, FactorEffect],
    model_id: str,
    gpu_name: str,
    backend: str,
    total_time_s: float,
    text_output_model: tuple[float, float, float, float] | None = None,
) -> str:
    """Generate human-readable summary report."""
    lines = []
    lines.append("=" * 70)
    lines.append("VLM INFERENCE SPEED BENCHMARK RESULTS")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Model: {model_id}")
    lines.append(f"GPU: {gpu_name}")
    lines.append(f"Backend: {backend}")
    lines.append(f"Total runs: {len(runs)}")
    lines.append(f"Benchmark duration: {total_time_s:.1f} seconds")
    lines.append("")
    
    # Factor effects
    lines.append("=" * 70)
    lines.append("FACTOR EFFECT ANALYSIS")
    lines.append("=" * 70)
    lines.append("")
    
    if "visual" in effects:
        e = effects["visual"]
        lines.append("Visual Token Effect (Sweep 1):")
        lines.append(f"  Coefficient: {e.coefficient:.4f} ms/token (±{e.std_error:.4f})")
        lines.append(f"  R²: {e.r_squared:.4f}")
        lines.append(f"  Data points: {e.data_points}")
        lines.append(f"  Interpretation: Each 1000 visual tokens adds ~{e.coefficient * 1000:.1f} ms")
        lines.append("")
    
    if "text" in effects:
        e = effects["text"]
        lines.append("Text Token Effect (Sweep 2):")
        lines.append(f"  Coefficient: {e.coefficient:.4f} ms/token (±{e.std_error:.4f})")
        lines.append(f"  R²: {e.r_squared:.4f}")
        lines.append(f"  Data points: {e.data_points}")
        lines.append(f"  Interpretation: Each 1000 text tokens adds ~{e.coefficient * 1000:.1f} ms")
        lines.append("")
    
    if "output" in effects:
        e = effects["output"]
        lines.append("Output Token Effect (Sweep 3):")
        lines.append(f"  Coefficient: {e.coefficient:.4f} ms/token (±{e.std_error:.4f})")
        lines.append(f"  R²: {e.r_squared:.4f}")
        lines.append(f"  Data points: {e.data_points}")
        lines.append(f"  Interpretation: Each 1000 output tokens adds ~{e.coefficient * 1000:.1f} ms")
        lines.append("")
    
    # Full prediction model (with visual tokens)
    intercept, beta_v, beta_t, beta_o = compute_combined_model(effects)
    lines.append("=" * 70)
    lines.append("LATENCY PREDICTION MODELS")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Model 1: Full model (visual + text + output)")
    lines.append(f"  latency_ms = {intercept:.1f} + {beta_v:.4f}×visual + {beta_t:.4f}×text + {beta_o:.4f}×output")
    lines.append("")
    
    # Text+output only model (more practical, ignores visual)
    if text_output_model is not None:
        to_intercept, to_beta_text, to_beta_output, to_r2 = text_output_model
        lines.append("Model 2: Text + Output only (combined from individual sweeps)")
        lines.append(f"  latency_ms = {to_intercept:.1f} + {to_beta_text:.4f}×text + {to_beta_output:.4f}×output")
        lines.append(f"  R² (validation) = {to_r2:.4f}")
        lines.append("")
        lines.append("  Coefficients from individual sweeps; R² computed on combined data.")
        lines.append("  Use Model 2 if visual token effect is weak (low R² in visual sweep).")
        lines.append("")
    elif "text" in effects and "output" in effects:
        # Fallback: combine individual sweep coefficients
        text_e = effects["text"]
        output_e = effects["output"]
        simple_intercept = (text_e.intercept + output_e.intercept) / 2
        lines.append("Model 2: Text + Output only (combined from individual sweeps)")
        lines.append(f"  latency_ms = {simple_intercept:.1f} + {text_e.coefficient:.4f}×text + {output_e.coefficient:.4f}×output")
        lines.append("")
        lines.append("  Use Model 2 if visual token effect is weak (low R²).")
        lines.append("")
    
    # Example predictions
    lines.append("-" * 70)
    lines.append("Example predictions (using Model 1):")
    lines.append("")
    
    # Typical config
    v, t, o = 588, 3000, 750  # User's reported config
    predicted = intercept + beta_v * v + beta_t * t + beta_o * o
    lines.append("  Typical (588 visual, 3000 text, 750 output):")
    lines.append(f"    Predicted: {predicted:.0f} ms ({predicted/1000:.1f} seconds)")
    lines.append("")
    
    # Small config
    v, t, o = 256, 500, 100
    predicted = intercept + beta_v * v + beta_t * t + beta_o * o
    lines.append("  Small (256 visual, 500 text, 100 output):")
    lines.append(f"    Predicted: {predicted:.0f} ms ({predicted/1000:.1f} seconds)")
    lines.append("")
    
    # Large config
    v, t, o = 2000, 3500, 1000
    predicted = intercept + beta_v * v + beta_t * t + beta_o * o
    lines.append("  Large (2000 visual, 3500 text, 1000 output):")
    lines.append(f"    Predicted: {predicted:.0f} ms ({predicted/1000:.1f} seconds)")
    lines.append("")
    
    # Model 2 predictions
    if text_output_model is not None:
        to_intercept, to_beta_text, to_beta_output, _ = text_output_model
        
        lines.append("-" * 70)
        lines.append("Example predictions (using Model 2 - text + output only):")
        lines.append("")
        
        t, o = 3000, 750
        predicted2 = to_intercept + to_beta_text * t + to_beta_output * o
        lines.append("  Typical (3000 text, 750 output):")
        lines.append(f"    Predicted: {predicted2:.0f} ms ({predicted2/1000:.1f} seconds)")
        lines.append("")
        
        t, o = 500, 100
        predicted2 = to_intercept + to_beta_text * t + to_beta_output * o
        lines.append("  Small (500 text, 100 output):")
        lines.append(f"    Predicted: {predicted2:.0f} ms ({predicted2/1000:.1f} seconds)")
        lines.append("")
        
        t, o = 3500, 1000
        predicted2 = to_intercept + to_beta_text * t + to_beta_output * o
        lines.append("  Large (3500 text, 1000 output):")
        lines.append(f"    Predicted: {predicted2:.0f} ms ({predicted2/1000:.1f} seconds)")
        lines.append("")
    elif "text" in effects and "output" in effects:
        text_e = effects["text"]
        output_e = effects["output"]
        simple_intercept = (text_e.intercept + output_e.intercept) / 2
        
        lines.append("-" * 70)
        lines.append("Example predictions (using Model 2 - text + output only):")
        lines.append("")
        
        t, o = 3000, 750
        predicted2 = simple_intercept + text_e.coefficient * t + output_e.coefficient * o
        lines.append("  Typical (3000 text, 750 output):")
        lines.append(f"    Predicted: {predicted2:.0f} ms ({predicted2/1000:.1f} seconds)")
        lines.append("")
        
        t, o = 500, 100
        predicted2 = simple_intercept + text_e.coefficient * t + output_e.coefficient * o
        lines.append("  Small (500 text, 100 output):")
        lines.append(f"    Predicted: {predicted2:.0f} ms ({predicted2/1000:.1f} seconds)")
        lines.append("")
        
        t, o = 3500, 1000
        predicted2 = simple_intercept + text_e.coefficient * t + output_e.coefficient * o
        lines.append("  Large (3500 text, 1000 output):")
        lines.append(f"    Predicted: {predicted2:.0f} ms ({predicted2/1000:.1f} seconds)")
        lines.append("")
    
    # Raw data table
    lines.append("=" * 70)
    lines.append("RAW DATA SUMMARY (by configuration)")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"{'Sweep':<8} {'Image':<12} {'Attrs':<6} {'MaxTok':<7} "
                f"{'VisualT':<8} {'TextT':<7} {'OutT':<6} {'Latency(ms)':<12} {'Memory(GB)':<10}")
    lines.append("-" * 90)
    
    # Group runs by config and compute averages
    from collections import defaultdict
    config_stats = defaultdict(list)
    
    for r in runs:
        key = (r.sweep_type, r.image_size, r.num_attributes, r.max_new_tokens)
        config_stats[key].append(r)
    
    for key in sorted(config_stats.keys()):
        sweep, img_size, n_attrs, max_tok = key
        runs_for_config = config_stats[key]
        
        avg_visual = np.mean([r.visual_tokens for r in runs_for_config])
        avg_text = np.mean([r.text_tokens for r in runs_for_config])
        avg_output = np.mean([r.output_tokens for r in runs_for_config])
        avg_latency = np.mean([r.total_ms for r in runs_for_config])
        avg_memory = np.mean([r.peak_memory_gb for r in runs_for_config])
        
        img_str = f"{img_size[0]}x{img_size[1]}"
        lines.append(f"{sweep:<8} {img_str:<12} {n_attrs:<6} {max_tok:<7} "
                    f"{avg_visual:<8.0f} {avg_text:<7.0f} {avg_output:<6.0f} "
                    f"{avg_latency:<12.1f} {avg_memory:<10.2f}")
    
    lines.append("")
    return "\n".join(lines)


def generate_plots(
    runs: list[BenchmarkRun],
    effects: dict[str, FactorEffect],
    output_dir: Path,
):
    """Generate visualization plots (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warning] matplotlib not installed, skipping plots")
        return
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Visual tokens effect plot
    visual_runs = [r for r in runs if r.sweep_type == "visual"]
    if visual_runs and "visual" in effects:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.array([r.visual_tokens for r in visual_runs])
        y = np.array([r.total_ms for r in visual_runs])
        
        ax.scatter(x, y, alpha=0.6, label="Measurements")
        
        # Regression line
        e = effects["visual"]
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = e.intercept + e.coefficient * x_line
        ax.plot(x_line, y_line, 'r-', linewidth=2, 
                label=f"Fit: {e.coefficient:.4f} ms/token (R²={e.r_squared:.3f})")
        
        ax.set_xlabel("Visual Tokens")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Effect of Visual Tokens on Inference Latency")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "visual_tokens_effect.png", dpi=150)
        plt.close()
        print(f"[Plot] Saved: {plots_dir / 'visual_tokens_effect.png'}")
    
    # Text tokens effect plot
    text_runs = [r for r in runs if r.sweep_type == "text"]
    if text_runs and "text" in effects:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.array([r.text_tokens for r in text_runs])
        y = np.array([r.total_ms for r in text_runs])
        
        ax.scatter(x, y, alpha=0.6, label="Measurements")
        
        e = effects["text"]
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = e.intercept + e.coefficient * x_line
        ax.plot(x_line, y_line, 'r-', linewidth=2,
                label=f"Fit: {e.coefficient:.4f} ms/token (R²={e.r_squared:.3f})")
        
        ax.set_xlabel("Text Tokens")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Effect of Text Tokens on Inference Latency")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "text_tokens_effect.png", dpi=150)
        plt.close()
        print(f"[Plot] Saved: {plots_dir / 'text_tokens_effect.png'}")
    
    # Output tokens effect plot
    output_runs = [r for r in runs if r.sweep_type == "output"]
    if output_runs and "output" in effects:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.array([r.output_tokens for r in output_runs])
        y = np.array([r.total_ms for r in output_runs])
        
        ax.scatter(x, y, alpha=0.6, label="Measurements")
        
        e = effects["output"]
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = e.intercept + e.coefficient * x_line
        ax.plot(x_line, y_line, 'r-', linewidth=2,
                label=f"Fit: {e.coefficient:.4f} ms/token (R²={e.r_squared:.3f})")
        
        ax.set_xlabel("Output Tokens")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Effect of Output Tokens on Inference Latency")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "output_tokens_effect.png", dpi=150)
        plt.close()
        print(f"[Plot] Saved: {plots_dir / 'output_tokens_effect.png'}")
    
    # Combined overview plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (sweep_type, ax, xlabel) in enumerate([
        ("visual", axes[0], "Visual Tokens"),
        ("text", axes[1], "Text Tokens"),
        ("output", axes[2], "Output Tokens"),
    ]):
        sweep_runs = [r for r in runs if r.sweep_type == sweep_type]
        if sweep_runs and sweep_type in effects:
            if sweep_type == "visual":
                x = np.array([r.visual_tokens for r in sweep_runs])
            elif sweep_type == "text":
                x = np.array([r.text_tokens for r in sweep_runs])
            else:
                x = np.array([r.output_tokens for r in sweep_runs])
            
            y = np.array([r.total_ms for r in sweep_runs])
            
            ax.scatter(x, y, alpha=0.6)
            
            e = effects[sweep_type]
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = e.intercept + e.coefficient * x_line
            ax.plot(x_line, y_line, 'r-', linewidth=2)
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Latency (ms)")
            ax.set_title(f"{e.coefficient:.4f} ms/token\n(R²={e.r_squared:.3f})")
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "factor_effects_overview.png", dpi=150)
    plt.close()
    print(f"[Plot] Saved: {plots_dir / 'factor_effects_overview.png'}")


# =============================================================================
# Main Entry Points
# =============================================================================

def run_benchmark(
    preset: str = "quick",
    model_id: str = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
    backend: str = "vllm",
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 8192,
    output_dir: str = "benchmark_results",
    generate_plots_flag: bool = False,
    debug: bool = False,
) -> tuple[list[BenchmarkRun], dict[str, FactorEffect]]:
    """Run the complete benchmark."""
    
    # Get preset configuration
    if preset == "quick":
        sweeps = get_quick_preset()
    elif preset == "full":
        sweeps = get_full_preset()
    else:
        raise ValueError(f"Unknown preset: {preset}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize benchmark
    benchmark = VLMBenchmark(
        model_id=model_id,
        backend=backend,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        output_dir=output_dir,
        debug=debug,
    )
    
    # Run all sweeps
    all_runs = []
    start_time = time.time()
    
    for sweep_name, sweep_config in sweeps.items():
        print(f"\n{'=' * 60}")
        print(f"STARTING SWEEP: {sweep_name.upper()}")
        print(f"{'=' * 60}")
        
        runs = benchmark.run_sweep(sweep_config)
        all_runs.extend(runs)
        
        # Save intermediate results
        intermediate_file = output_dir / f"runs_{sweep_name}.json"
        with open(intermediate_file, "w") as f:
            json.dump([asdict(r) for r in runs], f, indent=2)
        print(f"[Saved] {intermediate_file}")
    
    total_time = time.time() - start_time
    
    # Analyze results
    print(f"\n{'=' * 60}")
    print("ANALYZING RESULTS")
    print(f"{'=' * 60}")
    
    effects = analyze_factor_effects(all_runs)
    
    # Compute text+output only model (using individual sweep coefficients)
    text_output_model = compute_text_output_model(all_runs, effects)
    
    # Get GPU info
    gpu_name = benchmark._get_gpu_info()
    
    # Generate report
    report = generate_summary_report(
        all_runs, effects, model_id, gpu_name, backend, total_time,
        text_output_model=text_output_model,
    )
    print("\n" + report)
    
    # Save results
    results_file = output_dir / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump([asdict(r) for r in all_runs], f, indent=2)
    print(f"[Saved] {results_file}")
    
    # Save factor analysis
    analysis_file = output_dir / "factor_analysis.json"
    with open(analysis_file, "w") as f:
        json.dump({k: asdict(v) for k, v in effects.items()}, f, indent=2)
    print(f"[Saved] {analysis_file}")
    
    # Save summary report
    summary_file = output_dir / "summary.txt"
    with open(summary_file, "w") as f:
        f.write(report)
    print(f"[Saved] {summary_file}")
    
    # Generate plots
    if generate_plots_flag:
        generate_plots(all_runs, effects, output_dir)
    
    return all_runs, effects


def analyze_existing_results(
    results_file: str,
    output_dir: str | None = None,
    generate_plots_flag: bool = False,
) -> tuple[list[BenchmarkRun], dict[str, FactorEffect]]:
    """Analyze existing benchmark results."""
    
    results_path = Path(results_file)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, "r") as f:
        data = json.load(f)
    
    # Convert back to BenchmarkRun objects
    runs = []
    for item in data:
        # Handle tuple conversion for image_size
        if isinstance(item.get("image_size"), list):
            item["image_size"] = tuple(item["image_size"])
        runs.append(BenchmarkRun(**item))
    
    # Analyze
    effects = analyze_factor_effects(runs)
    
    # Compute text+output only model (using individual sweep coefficients)
    text_output_model = compute_text_output_model(runs, effects)
    
    # Generate report
    # Try to infer model/gpu info from file path or use defaults
    model_id = "Unknown"
    gpu_name = "Unknown"
    backend = "Unknown"
    
    report = generate_summary_report(
        runs, effects, model_id, gpu_name, backend, 0.0,
        text_output_model=text_output_model,
    )
    print("\n" + report)
    
    # Save analysis if output_dir specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        analysis_file = output_path / "factor_analysis.json"
        with open(analysis_file, "w") as f:
            json.dump({k: asdict(v) for k, v in effects.items()}, f, indent=2)
        print(f"[Saved] {analysis_file}")
        
        summary_file = output_path / "summary.txt"
        with open(summary_file, "w") as f:
            f.write(report)
        print(f"[Saved] {summary_file}")
        
        if generate_plots_flag:
            generate_plots(runs, effects, output_path)
    
    return runs, effects


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="VLM Inference Speed Benchmark for Qwen3-VL models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--preset",
        choices=["quick", "full"],
        default="quick",
        help="Benchmark preset: 'quick' (~20 min) or 'full' (~1.5 hours). Default: quick",
    )
    mode_group.add_argument(
        "--analyze-only",
        type=str,
        metavar="FILE",
        help="Analyze existing results file instead of running benchmark",
    )
    
    # Model configuration
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
        help="HuggingFace model ID (default: Qwen/Qwen3-VL-30B-A3B-Instruct-FP8)",
    )
    parser.add_argument(
        "--backend",
        choices=["vllm", "hf"],
        default="vllm",
        help="Inference backend: 'vllm' or 'hf' (HuggingFace). Default: vllm",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="GPU memory fraction for vLLM (0.0-1.0). Default: 0.90",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Max sequence length for vLLM. Default: 8192",
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Output directory for results. Default: benchmark_results",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate visualization plots (requires matplotlib)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )
    
    args = parser.parse_args()
    
    if args.analyze_only:
        # Analyze existing results
        print(f"Analyzing existing results: {args.analyze_only}")
        analyze_existing_results(
            args.analyze_only,
            output_dir=args.output_dir,
            generate_plots_flag=args.plot,
        )
    else:
        # Run benchmark
        print(f"Running {args.preset} benchmark...")
        run_benchmark(
            preset=args.preset,
            model_id=args.model_id,
            backend=args.backend,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            output_dir=args.output_dir,
            generate_plots_flag=args.plot,
            debug=args.debug,
        )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
