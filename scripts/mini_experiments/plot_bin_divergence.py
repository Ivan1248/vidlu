import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def kl(pred, target):
    return F.kl_div(pred.log().unsqueeze(0), target.unsqueeze(0), reduction='none').sum(1).squeeze(
        0)


def js(a, b):
    m = (a + b) / 2
    return (kl(a, m) + kl(b, m)) / 2


def rkl(pred, target):
    return kl(target, pred)


def sqr_log(a, b):
    return (a.log() - b.log()).pow(2).mean(0)


if __name__ == '__main__':
    resolution = 100
    margin = 0.01
    pred, targ = torch.tensor(
        np.mgrid[:resolution + 1, :resolution + 1]) * ((1 - 2 * margin) / (resolution)) + margin
    pred = torch.stack([pred, 1 - pred])
    targ = torch.stack([targ, 1 - targ])

    fig, ax = plt.subplots(nrows=2, ncols=2)

    divergences = [kl, js, rkl, sqr_log]
    titles = ['KL div.', 'JS div.', 'Rev. KL div.', 'Squared log']

    for i, divergence in enumerate(divergences):
        div_mat = divergence(pred, targ)
        der_div_mat = (div_mat[:, 1:] - div_mat[:, :-1]) * resolution * (1 - 2 * margin)
        der_div_mat = der_div_mat.abs()
        #der_div_mat = div_mat

        im = ax[i // 2, i % 2].imshow(der_div_mat, extent=(margin, 1 - margin, margin, 1 - margin),
                                      cmap='inferno')
        ax[i // 2, i % 2].set_title(titles[i])
        ax[i // 2, i % 2].set_xlabel('Predicted probability')
        ax[i // 2, i % 2].set_ylabel('Target probability')
        fig.colorbar(im, ax=ax[i // 2, i % 2])

    fig.suptitle('Absolute derivative with respect to the predicted probability')

    plt.tight_layout()
    plt.show()
