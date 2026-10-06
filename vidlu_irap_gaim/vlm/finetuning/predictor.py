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
from typing import Any, Mapping, Sequence

import irap_evaluation as ie
from PIL import Image

from vidlu_irap_gaim.vlm.models.base import BaseVLMPredictor
from vidlu_irap_gaim.vlm.models.generation import warn_if_truncated
from vidlu_irap_gaim.vlm.response_scheme import DEFAULT_RESPONSE_TOKEN_MARGIN
from .model import _BaseVLMClassifier
from .loading import load_base_classifier
from vidlu_irap_gaim.tools.vlm_inference import run_evaluation_on_data_entries


class VLMClassifierPredictor(BaseVLMPredictor):
    """Predictor wrapping a loaded `_BaseVLMClassifier` for evaluation.

    Serves the pretrained model and a fine-tuned one alike -- `use_lora=False`
    (what `load_base_classifier` passes) simply means there is no adapter -- so
    the name says "classifier", not "fine-tuned".

    Usage:
        predictor = VLMClassifierPredictor(trainer.model)
        result = run_evaluation(test_dataset, predictor, output_dir, model_info)

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
        """Responds to ``prompt`` for every image, batched by the classifier.

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


def run_zero_shot_eval(experiment, split_prefix: str = "test", *, model_id: str | None = None,
                       load_in_4bit: bool | None = None, output_subdir: str = "vlm_zero_shot_eval",
                       method_name: str | None = None, seed: int | None = None, **kwargs):
    """Evaluates the *pretrained* model on the experiment's datasets.

    The counterpart of `run_full_eval`: same datasets, same prompts, same
    generation path, no LoRA adapter. Defaults to the experiment model's own
    `model_id` and quantization so that the only difference between the two runs
    is the adapter -- the comparison is otherwise not attributable to it.

    Called from run.py test with no `-r`, since there is no checkpoint to load::

        python scripts/run.py test ... -m "vidlu_irap_gaim.vlm.finetuning.predictor:run_zero_shot_eval(e,attrs_per_session=1)"

    Args:
        experiment: TrainingExperiment instance from Vidlu.
        split_prefix: Prefix for splits to evaluate ("test", "val").
        model_id: Base model to load. None uses the experiment model's.
        load_in_4bit: None uses the experiment model's setting.
        output_subdir: Subdirectory under the experiment dir for the results.
        method_name: The method name in the prediction files. Defaults to the model ID.
        seed: The model's seed in the prediction files.
        **kwargs: Passed to `run_eval` (attrs_per_session, ...).
    """
    trained = experiment.trainer.model
    classifier = load_base_classifier(
        model_id if model_id is not None else trained.model_id,
        classifier_class=type(trained),
        device=next(trained.parameters()).device,
        load_in_4bit=trained.load_in_4bit if load_in_4bit is None else load_in_4bit,
        enable_thinking=trained.enable_thinking,
    )
    model_info = ie.ModelInfo(method_name=method_name or classifier.model_id,
                              training_splits=(), early_stopping_splits=(), seed=seed)
    return run_eval(classifier, experiment.data, split_prefix,
                    experiment.cpman.experiment_dir / output_subdir, model_info, **kwargs)


def run_full_eval(experiment, split_prefix: str = "test", *,
                  output_subdir: str = "vlm_full_eval", method_name: str | None = None,
                  seed: int | None = None, **kwargs):
    """Runs full generative evaluation on the experiment's (fine-tuned) model.

    Called from run.py test, with `-r best` to load the fine-tuned checkpoint::

        python scripts/run.py test ... -r best -m "vidlu_irap_gaim.vlm.finetuning.predictor:run_full_eval(e,attrs_per_session=1)"

    The model in the prediction files is described by `make_model_info_from_experiment`.

    Args:
        experiment: TrainingExperiment instance from Vidlu.
        split_prefix: Prefix for splits to evaluate ("test", "val").
        output_subdir: Subdirectory under the experiment dir for the results.
        method_name, seed: See `make_model_info_from_experiment`.
        **kwargs: Passed to `run_eval` (attrs_per_session, ...).
    """
    # Imported here so that importing the package does not import `vidlu.experiments`.
    from vidlu_irap_gaim.prediction_files import make_model_info_from_experiment

    model_info = make_model_info_from_experiment(experiment, method_name, seed)
    return run_eval(experiment.trainer.model, experiment.data, split_prefix,
                    experiment.cpman.experiment_dir / output_subdir, model_info, **kwargs)


def run_eval(
        classifier: _BaseVLMClassifier,
        data: Mapping[str, Any],
        split_prefix: str,
        output_dir: str | Path,
        model_info: ie.ModelInfo,
        *,
        attrs_per_session: int | None = None,
        max_response_tokens: int | None = None,
        response_token_margin: int = DEFAULT_RESPONSE_TOKEN_MARGIN,
        amp: bool = False,
        min_new_tokens: int = 0,
        debug: bool = False,
        **kwargs,
):
    """Generative evaluation of one classifier on the data entries whose keys start with
    `split_prefix`, with the results of an entry in `output_dir / key`.

    Shared by `run_zero_shot_eval` and `run_full_eval` so that the pretrained and
    the fine-tuned run differ in nothing but the weights.

    Args:
        classifier: Loaded classifier to evaluate.
        data: The data entries by key, e.g. an experiment's `data`.
        split_prefix: Prefix of the keys of the data entries to evaluate ("test", "val").
        output_dir: Directory of the results.
        model_info: The classifier in the prediction files.
        attrs_per_session: Attributes per VLM session; None puts them all in one,
            1 gives each attribute its own session and its own prompt.
        max_response_tokens: None derives a per-session bound from the response
            scheme, which is what makes the budget scale with the session size.
        response_token_margin: Absolute tokens added to a derived budget.
        amp: Autocast generation to bfloat16.
        debug: Print detailed prompt and response debugging information.
        **kwargs: Passed to `run_evaluation` (batch_size, limit, ...).

    Returns:
        The results by data entry key.
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
    return run_evaluation_on_data_entries(data, split_prefix, predictor, output_dir, model_info,
                                          **kwargs)
