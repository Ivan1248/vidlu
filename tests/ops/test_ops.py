import torch

from vidlu.torch_utils import save_grads


class TestSaveGrads:
    def test_save_grads(self):
        x = torch.tensor([2.], requires_grad=True)
        (x ** 2).backward()

        pre_grad = x.grad.detach().clone()
        with save_grads([x]):
            for _ in range(5):
                (x ** 2).backward()
                assert torch.all(x.grad != pre_grad)
        assert torch.all(x.grad == pre_grad)
