import torch

from vidlu.data import Record


# Extend output


def classification_extend_output(output):
    logits = output[0] if isinstance(output, tuple) else output
    if not isinstance(logits, torch.Tensor):
        raise ValueError("The output or its first element must be a `torch.Tensor`"
                         + " representing class scores.")
    return Record(output=logits, full_output=output, log_probs_=lambda: logits.log_softmax(1),
                  probs_=lambda r: r.log_probs.exp(), hard_prediction_=lambda: logits.argmax(1))

