import torch
import vidlu.modules.losses as vml
import vidlu.ops as vo

torch.no_grad()


class TestLosses:
    def test_entropy(self):
        logits = torch.randn(2, 3, 4, 5)
        probs = logits.softmax(1)
        ent1 = vml.entropy(logits)
        ent2 = vml.cross_entropy_loss_with_logits(logits, probs)
        assert torch.all(ent1.sub(ent2).abs() < 1e-6)

    def test_cross_entropy(self):
        C = 3
        logits = torch.randn(2, C, 4, 5)
        labels = torch.randint(C, (2, 4, 5))
        labels_oh = vo.one_hot(labels, C).permute(0, 3, 1, 2)
        nll = vml.nll_loss_with_logits(logits, labels)
        ce = vml.cross_entropy_loss_with_logits(logits, labels_oh)
        assert torch.all(ce - nll < 1e-6)
