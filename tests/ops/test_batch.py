import torch
import numpy as np

from vidlu.ops import batch

torch.no_grad()

weaker_tols = dict(rtol=1e-3, atol=1e-4)


class TestBatchops:
    def test_norm(self):  # in PyTorch 1.1.0 the test doesn't pass for float32
        shapes = [(4, 1), (5, 3, 7), (1, 100, 100), (2, 1000, 1000)]
        ps = [0, 0.5, 1, 2, 7, 11, np.inf]
        for p in ps:
            for s in shapes:
                x = torch.zeros(s, dtype=torch.float64).uniform_(-1, 2)
                x /= x.numel() ** (1 / (p + 1e-1))
                norms = batch.norm(x, p)
                for i, e in enumerate(x):  # bug in torch.norm in pytorch <= 1.0.0
                    assert torch.allclose(norms[i], torch.norm(e.view(-1), p, dim=0))

    def test_project_to_1_ball(self):
        shapes = [(5, 5), (5, 100, 100), (4, 1000, 1000)]
        target_max_norms = [0.001, 1, 1000]
        for s in shapes:
            for n in target_max_norms:
                x = torch.zeros(s).uniform_(-2, 2)
                x[:2] /= 2 * x[0].numel()
                pre_norms = x.view(x.shape[0], -1).abs().sum(1)
                x_proj = batch.project_to_1_ball(x, n)
                x_proj = batch.project_to_1_ball(x_proj, n)
                norms = x_proj.view(x.shape[0], -1).abs().sum(1)
                target_norms = torch.min(torch.full_like(norms, n), pre_norms)
                assert torch.allclose(norms, target_norms, **weaker_tols)
