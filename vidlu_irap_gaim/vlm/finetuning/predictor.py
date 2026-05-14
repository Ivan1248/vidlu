"""
Fine-tuned VLM predictor for evaluation using existing vlm_inference code.

This allows running proper evaluation with generation and parsing on
fine-tuned models, reusing the existing evaluation infrastructure.
"""

from pathlib import Path

import torch
from PIL import Image

from vidlu_irap_gaim.vlm.models.base import BaseVLMPredictor
from .model import Qwen3VLClassifier
from vidlu_irap_gaim.vlm.models.qwen_utils import build_qwen_chat_messages
from vidlu_irap_gaim.tools.vlm_inference import run_evaluation


class FineTunedVLMPredictor(BaseVLMPredictor):
    """Predictor wrapping a fine-tuned Qwen3VLClassifier for evaluation.

    This predictor allows using the existing vlm_inference.py evaluation
    infrastructure with fine-tuned models. It wraps a Qwen3VLClassifier
    and provides the same interface as other VLM predictors.

    Usage:
        # After training, create predictor from trained model
        predictor = FineTunedVLMPredictor(trainer.model)

        # Use with existing evaluation
        from vidlu_irap_gaim.tools.vlm_inference import run_evaluation
        result = run_evaluation(
            dataset=test_dataset,
            predictor=predictor,
            ...
        )

    Args:
        model: Fine-tuned Qwen3VLClassifier instance.
        max_response_tokens: Maximum tokens to generate.
        prompt_config_path: Optional path to prompt YAML config.
        chunk_size: Max attributes per VLM call.
        min_new_tokens: Minimum tokens to generate.
        debug: Enable debug output.
    """

    def __init__(
            self,
            model: "Qwen3VLClassifier",
            max_response_tokens: int = 512,
            prompt_config_path: str | Path | None = None,
            chunk_size: int = 15,
            min_new_tokens: int = 0,
            debug: bool = False,
    ):
        super().__init__(
            model_id=model.model_id,
            max_response_tokens=max_response_tokens,
            prompt_config_path=prompt_config_path,
            chunk_size=chunk_size,
            min_new_tokens=min_new_tokens,
            debug=debug,
        )
        self._ft_model = model

    def _load_model(self) -> None:
        """Load model - uses the already-loaded fine-tuned model."""
        self._ft_model._load()
        self._model = self._ft_model._model
        self._processor = self._ft_model._processor

    def _generate_single(self, pil_image: Image.Image, prompt: str) -> tuple[str, str | None]:
        """Generate response for a single prompt+image.

        Uses the same flow as zero-shot inference to ensure consistency
        between training and evaluation.

        Args:
            pil_image: Input image as PIL Image.
            prompt: Text prompt.

        Returns:
            Tuple of (response_text, thinking_text). thinking_text is always
            None for fine-tuned models (thinking is not supported).
        """
        from qwen_vl_utils import process_vision_info

        messages = build_qwen_chat_messages(pil_image, prompt)
        # Apply chat template
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process image through qwen_vl_utils (matches training flow)
        image_inputs, video_inputs = process_vision_info(messages)

        # Tokenize
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        if self.debug:
            print(f"[DEBUG] Input shape: {inputs.input_ids.shape}, Prompt chars: {len(prompt)}")

        # Generate
        with torch.no_grad():
            gen_kwargs = {"max_response_tokens": self.max_response_tokens}
            if self.min_new_tokens > 0:
                gen_kwargs["min_new_tokens"] = self.min_new_tokens

            generated_ids = self._model.generate(**inputs, **gen_kwargs)

        # Decode only newly generated tokens
        input_len = inputs.input_ids.shape[1]
        output_ids = generated_ids[:, input_len:]
        raw_response = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        if self.debug:
            print(
                f"[DEBUG] Output tokens: {output_ids.shape[1]}, Response: {raw_response[:200]}..."
            )

        return raw_response, None


def run_full_eval(e, split_prefix: str = "test", **kwargs):
    """Run full generative evaluation on a fine-tuned model.

    This function is designed to be called from run.py test -m:
        python scripts/run.py test ... -m "irap_gaim.vlm.finetuning:run_full_eval"

    Args:
        e: TrainingExperiment instance from Vidlu.
        split_prefix: Prefix for splits to evaluate ("test", "val").
        **kwargs: Additional arguments passed to run_evaluation.

    Returns:
        Dictionary mapping split names to evaluation results.
    """
    # Create predictor from the trained model
    predictor = FineTunedVLMPredictor(e.trainer.model)

    # Get output directory
    output_base = e.cpman.experiment_dir / "vlm_full_eval"

    # Evaluate on matching splits
    results = {}
    for name, ds in e.data.items():
        if name.startswith(split_prefix):
            print(f"\n{'=' * 60}")
            print(f"Full generative evaluation on: {name}")
            print(f"{'=' * 60}")

            result = run_evaluation(
                dataset=ds,
                predictor=predictor,
                split=name,
                output_dir=output_base / name,
                **kwargs,
            )
            results[name] = result

            if result.metrics:
                print(f"  amF1: {result.metrics.get('amF1', 'N/A'):.4f}")

    return results
