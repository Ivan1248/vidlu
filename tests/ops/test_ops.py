import pytest
import torch

from vidlu.torch_utils import preserve_grads


class TestSaveGrads:
    @pytest.mark.skip(reason="Fatal Python error: Aborted in GitHub Actions.")
    def test_save_grads(self):
        return
        x = torch.tensor([2.], requires_grad=True)
        (x ** 2).backward()

        pre_grad = x.grad.detach().clone()
        with preserve_grads([x]):
            for _ in range(5):
                (x ** 2).backward()
                assert torch.all(x.grad != pre_grad)
        assert torch.all(x.grad == pre_grad)
