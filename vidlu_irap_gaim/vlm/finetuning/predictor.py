"""
Predictor wrapping a `_BaseVLMClassifier` so the shared evaluation pipeline can drive it.

The classifier owns the chat template, the vision preprocessing and the
generation path, so evaluation during training and evaluation afterwards cannot
diverge. `load_base_classifier` / `load_finetuned_classifier` produce the
pretrained and the fine-tuned model through that same path, which is what makes
their scores comparable: same backend, same quantization, same prompts, and only
the LoRA adapter differing.
"""

from pathlib import Path
from typing import Sequence

from PIL import Image

from vidlu_irap_gaim.vlm.models.base import BaseVLMPredictor
from vidlu_irap_gaim.vlm.models.generation import warn_if_truncated
from vidlu_irap_gaim.vlm.response_scheme import DEFAULT_RESPONSE_TOKEN_MARGIN
from .model import _BaseVLMClassifier
from .loading import load_base_classifier
from vidlu_irap_gaim.tools.vlm_inference import run_evaluation


class VLMClassifierPredictor(BaseVLMPredictor):
    """Predictor wrapping a loaded `_BaseVLMClassifier` for evaluation.

    Serves the pretrained model and a fine-tuned one alike -- `use_lora=False`
    (what `load_base_classifier` passes) simply means there is no adapter -- so
    the name says "classifier", not "fine-tuned".

    Usage:
        predictor = VLMClassifierPredictor(trainer.model)
        result = run_evaluation(dataset=test_dataset, predictor=predictor, ...)

    Args:
        model: Loaded classifier (`Qwen3VLClassifier`, `Gemma4VLClassifier`, ...).
        max_response_tokens: Maximum response tokens per session; None derives a
            bound per session from the response scheme.
        response_scheme: Prompt/response convention. Leave None to take the
            evaluated dataset's, which is what keeps prompting and scoring on one
            convention.
        prompt_config_path: Optional path to prompt YAML config.
        attrs_per_session: Attributes per VLM session; None puts them all in one.
        response_token_margin: Absolute tokens added to a derived budget.
        min_new_tokens: Minimum tokens to generate.
        amp: Autocast generation to bfloat16.
        debug: Enable debug output.
    """

    def __init__(
            self,
            model: _BaseVLMClassifier,
            max_response_tokens: int | None = 512,
            response_scheme=None,
            prompt_config_path: str | Path | None = None,
            attrs_per_session: int | None = None,
            response_token_margin: int = DEFAULT_RESPONSE_TOKEN_MARGIN,
            min_new_tokens: int = 0,
            amp: bool = False,
            debug: bool = False,
    ):
        self._classifier = model
        # `enable_thinking` and `thinking_budget` are taken from the classifier, which renders
        # the chat template, splits the reasoning off and spends the reasoning budget
        # (`_generate_within_budget`); this predictor never consults its own copies.
        super().__init__(
            model_id=model.model_id,
            max_response_tokens=max_response_tokens,
            response_scheme=response_scheme,
            prompt_config_path=prompt_config_path,
            attrs_per_session=attrs_per_session,
            response_token_margin=response_token_margin,
            min_new_tokens=min_new_tokens,
            debug=debug,
            enable_thinking=model.enable_thinking,
            thinking_budget=model.thinking_budget,
        )
        self.amp = amp

    @property
    def tokenizer(self):
        return self._classifier.tokenizer

    def _load_model(self) -> None:
        """Loads the model - uses the already-loaded classifier."""
        self._classifier._load()
        self._model = self._classifier._model
        self._processor = self._classifier._processor

    def _generate_batch(
            self,
            pil_images: Sequence[Image.Image],
            prompt: str,
            max_response_tokens: int,
    ) -> list[tuple[str, str | None, bool | None]]:
        """Responses ``prompt`` for every image, batched by the classifier.

        The classifier's own generation path, not a copy of it: it owns the message format,
        the chat-template kwargs and the vision preprocessing (which differ per model family,
        so a copy would silently be Qwen-only), and the reasoning/response budget split. Eval
        during training and eval afterwards therefore cannot diverge.
        """
        results = self._classifier.generate_batch_for_eval(
            images=list(pil_images),
            prompts=[prompt] * len(pil_images),
            max_response_tokens=max_response_tokens,
            amp=self.amp,
            min_new_tokens=self.min_new_tokens,
        )

        warn_if_truncated([truncated for _, _, truncated in results], max_response_tokens)

        if self.debug and results:
            print(f"[DEBUG] Response: {results[0][0][:200]}...")

        return results


def run_zero_shot_eval(e, split_prefix: str = "test", *, model_id: str | None = None,
                       load_in_4bit: bool | None = None, output_subdir: str = "vlm_zero_shot_eval",
                       **kwargs):
    """Evaluates the *pretrained* model on the experiment's datasets.

    The counterpart of `run_full_eval`: same datasets, same prompts, same
    generation path, no LoRA adapter. Defaults to the experiment model's own
    `model_id` and quantization so that the only difference between the two runs
    is the adapter -- the comparison is otherwise not attributable to it.

    Called from run.py test with no `-r`, since there is no checkpoint to load::

        python scripts/run.py test ... -m "vidlu_irap_gaim.vlm.finetuning.predictor:run_zero_shot_eval,e,attrs_per_session=1"

    Args:
        e: TrainingExperiment instance from Vidlu.
        split_prefix: Prefix for splits to evaluate ("test", "val").
        model_id: Base model to load. None uses the experiment model's.
        load_in_4bit: None uses the experiment model's setting.
        output_subdir: Subdirectory under the experiment dir for the results.
        **kwargs: Passed to `run_eval` (attrs_per_session, ...).
    """
    trained = e.trainer.model
    classifier = load_base_classifier(
        model_id if model_id is not None else trained.model_id,
        classifier_class=type(trained),
        device=next(trained.parameters()).device,
        load_in_4bit=trained.load_in_4bit if load_in_4bit is None else load_in_4bit,
        enable_thinking=trained.enable_thinking,
    )
    return run_eval(classifier, e, split_prefix=split_prefix, output_subdir=output_subdir,
                    **kwargs)


def run_full_eval(e, split_prefix: str = "test", *, output_subdir: str = "vlm_full_eval",
                  **kwargs):
    """Runs full generative evaluation on the experiment's (fine-tuned) model.

    Called from run.py test, with `-r best` to load the fine-tuned checkpoint::

        python scripts/run.py test ... -r best -m "vidlu_irap_gaim.vlm.finetuning.predictor:run_full_eval,e,attrs_per_session=1"

    Args:
        e: TrainingExperiment instance from Vidlu.
        split_prefix: Prefix for splits to evaluate ("test", "val").
        output_subdir: Subdirectory under the experiment dir for the results.
        **kwargs: Passed to `run_eval` (attrs_per_session, ...).
    """
    return run_eval(e.trainer.model, e, split_prefix=split_prefix, output_subdir=output_subdir,
                    **kwargs)


def run_eval(
        classifier: _BaseVLMClassifier,
        e,
        *,
        split_prefix: str = "test",
        output_subdir: str = "vlm_full_eval",
        attrs_per_session: int | None = None,
        max_response_tokens: int | None = None,
        response_token_margin: int = DEFAULT_RESPONSE_TOKEN_MARGIN,
        batch_size: int | None = None,
        batch_tokens: int = 16384,
        amp: bool = False,
        min_new_tokens: int = 0,
        debug: bool = False,
        **kwargs,
):
    """Generative evaluation of one classifier over the experiment's splits.

    Shared by `run_zero_shot_eval` and `run_full_eval` so that the pretrained and
    the fine-tuned run differ in nothing but the weights.

    Args:
        classifier: Loaded classifier to evaluate.
        e: TrainingExperiment instance from Vidlu.
        split_prefix: Prefix for splits to evaluate ("test", "val").
        output_subdir: Subdirectory under the experiment dir for the results.
        attrs_per_session: Attributes per VLM session; None puts them all in one,
            1 gives each attribute its own session and its own prompt.
        max_response_tokens: None derives a per-session bound from the response
            scheme, which is what makes the budget scale with the session size.
        response_token_margin: Absolute tokens added to a derived budget.
        batch_size: Images generated for at once; None sizes it from `batch_tokens`.
        batch_tokens: Target prompt tokens per batch, used when `batch_size` is None.
        amp: Autocast generation to bfloat16.
        **kwargs: Further arguments for `run_evaluation`.

    Returns:
        Dictionary mapping split names to evaluation results.
    """
    predictor = VLMClassifierPredictor(
        classifier,
        max_response_tokens=max_response_tokens,
        attrs_per_session=attrs_per_session,
        response_token_margin=response_token_margin,
        min_new_tokens=min_new_tokens,
        amp=amp,
        debug=debug,
    )

    output_base = e.cpman.experiment_dir / output_subdir

    results = {}
    for name, ds in e.data.items():
        if name.startswith(split_prefix):
            print(f"\n{'=' * 60}")
            print(f"Generative evaluation on: {name}")
            print(f"{'=' * 60}")

            result = run_evaluation(
                dataset=ds,
                predictor=predictor,
                split=name,
                output_dir=output_base / name,
                batch_size=batch_size,
                batch_tokens=batch_tokens,
                debug=debug,
                **kwargs,
            )
            results[name] = result

            if result.metrics:
                print(f"  amF1: {result.metrics.get('amF1', 'N/A'):.4f}")

    return results
