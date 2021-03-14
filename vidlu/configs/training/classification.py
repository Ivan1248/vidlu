import torch

from vidlu.data import Record


# Extend output


def classification_extend_output(output):
    logits = output[0] if isinstance(output, tuple) else output
    if not isinstance(logits, torch.Tensor):
        raise ValueError("The output or its first element must be a `torch.Tensor`"
                         + " representing class scores.")
    return Record(output=logits, probs_=lambda: logits.softmax(1),
                  hard_prediction_=lambda: logits.argmax(1))
