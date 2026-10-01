import pytest
import torch
import vidlu.modules.losses as vml
import vidlu.ops as vo

torch.no_grad()


class TestLosses:
    def test_entropy(self):
        logits = torch.randn(2, 3, 4, 5)
        probs = logits.softmax(1)
        ent1 = vml.entropy_l(logits)
        ent2 = vml.crossentropy_l(logits, probs)
        assert torch.all(ent1.sub(ent2).abs() < 1e-6)

    def test_cross_entropy(self):
        C = 3
        logits = torch.randn(2, C, 4, 5)
        labels = torch.randint(C, (2, 4, 5))
        labels_oh = vo.one_hot(labels, C).permute(0, 3, 1, 2)
        nll = vml.nll_loss_l(logits, labels)
        ce = vml.crossentropy_l(logits, labels_oh)
        assert torch.all(ce - nll < 1e-6)

    def test_neg_soft_mIoU_of_perfect_predictions_is_minus_one(self):
        labels = torch.randint(3, (2, 4, 5))
        logits = 50 * vo.one_hot(labels, 3).permute(0, 3, 1, 2).float()
        for target in (labels, logits.softmax(1)):
            assert vml.neg_soft_mIoU_l(logits, target).item() == pytest.approx(-1)
            torch.testing.assert_close(vml.neg_soft_mIoU_l(logits, target, is_batch=True),
                                       torch.full((2,), -1.))

    def test_neg_soft_mIoU_of_labels_equals_that_of_one_hot_probabilities(self):
        logits = torch.randn(2, 3, 4, 5)
        labels = torch.randint(3, (2, 4, 5))
        labels_oh = vo.one_hot(labels, 3).permute(0, 3, 1, 2).float()
        for is_batch in (False, True):
            torch.testing.assert_close(vml.neg_soft_mIoU_l(logits, labels, is_batch=is_batch),
                                       vml.neg_soft_mIoU_l(logits, labels_oh, is_batch=is_batch))

    def test_neg_soft_mIoU_with_is_batch_is_computed_per_example(self):
        logits = torch.randn(2, 3, 4, 5)
        labels = torch.randint(3, (2, 4, 5))
        per_example = torch.stack([vml.neg_soft_mIoU_l(logits[i:i + 1], labels[i:i + 1])
                                   for i in range(2)])
        torch.testing.assert_close(vml.neg_soft_mIoU_l(logits, labels, is_batch=True),
                                   per_example)

    def test_neg_soft_mIoU_with_uniform_weights_equals_the_mean(self):
        logits = torch.randn(2, 3, 4, 5)
        labels = torch.randint(3, (2, 4, 5))
        weights = torch.full((3,), 1 / 3)
        for is_batch in (False, True):
            torch.testing.assert_close(
                vml.neg_soft_mIoU_l(logits, labels, is_batch=is_batch, weights=weights),
                vml.neg_soft_mIoU_l(logits, labels, is_batch=is_batch))
