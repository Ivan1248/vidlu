import torch
from torch import nn

celoss = nn.CrossEntropyLoss(ignore_index=-1)

logits = torch.randn(5, 10)
labels = torch.arange(5) - 1
print(celoss(logits, labels))
print(celoss(logits[1:], labels[1:]))
