import math
import argparse
from pathlib import Path
from functools import partial

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib

FORMAT = 'pdf'
SMALL_PDFS = True
matplotlib.rcParams["figure.dpi"] = 400

try:
    from vidlu.utils.presentation import figstyle

    get_figsize = figstyle.configure(document_fontsize=7 if SMALL_PDFS else 10)
except ImportError as e:
    print(f'Could not set up figure style: {e}')
    get_figsize = None

if SMALL_PDFS:
    matplotlib.use('pgf')


class TriClusterDataset:
    class_count = 2
    dstep = 1.5
    a = -10 * dstep
    c = 10 * dstep

    @classmethod
    def get_data(cls):
        a = cls.a
        c = cls.c
        dstep = cls.dstep
        data = [
            (a + 1 * dstep, None),
            (a + 2 * dstep, 0),
            (a + 3 * dstep, None),
            (a + 4 * dstep, None),

            (a + 7 * dstep, None),
            (a + 8 * dstep, None),
            (a + 9 * dstep, 1),
            (a + 10 * dstep, None),
            (a + 11 * dstep, None),
            (a + 12 * dstep, None),
            (a + 13 * dstep, None),

            (c - 4 * dstep, None),
            (c - 3 * dstep, 0),
            (c - 2 * dstep, None),
            (c - 1 * dstep, None),
            # (c + 0 * dstep, None),
            # (c + 4, None),
        ]

        data_l = [(x, y) for x, y in data if y is not None]
        x_l, y_l = tuple(map(np.array, zip(*data_l)))
        x_l = np.array(x_l, dtype=np.float32)
        x_u = np.array([x for x, y in data if y is None], dtype=np.float32)

        return x_l[:, None], y_l, x_u[:, None]

    @classmethod
    def get_uniform_data(cls):
        dstep = cls.dstep
        start = cls.a - 2 * dstep
        stop = cls.c + 2 * dstep
        x_u = np.linspace(start=start, stop=stop,
                          num=int((stop - start) / dstep + 0.5) + 1,
                          dtype=np.float32)
        return x_u[:, None]


def normal_kernel(a, b, std=1.):
    ak = a[:, None, :] / std  # (B, 1, C)
    kb = b[None, :, :] / std  # (1, N, C)
    return (ak - kb).pow(2).sum(-1).mul(-0.5).exp().div(std * (2 * math.pi) ** 0.5)  # (B, N)


class SemisupKernelClassifier(nn.Module):
    def __init__(self, x_l, y_l, x_u, num_classes=None, kernel=normal_kernel):
        super().__init__()
        if num_classes is None:
            num_classes = torch.max(y_l) + 1 if len(y_l) > 0 else 2
        self.x_a = torch.cat([x_l, x_u], dim=0) if len(x_l) > 0 else x_u
        self.y_l = y_l
        self.x_l, self.x_u = self.x_a[:len(y_l)], self.x_a[len(y_l):]
        self.kernel = kernel
        # s_u = torch.zeros((len(self.x_u), num_classes))
        # s_u[:, 0] = -5
        self.s_u = torch.nn.Parameter(torch.zeros((len(self.x_u), num_classes)))  # + s_u)

    def density(self, x, support=None):
        """Computes the density in the input.

        Args:
            x: input

        Returns:
            density p(x)
        """
        if support is None:
            support = self.x_a
        K = self.kernel(x, support)
        return K.sum(-1) / len(support)

    def parameter_values(self):
        pyl = F.one_hot(self.y_l.to(torch.long), self.s_u.shape[1])
        pyu = F.softmax(self.s_u, -1)
        return pyl, pyu

    def forward(self, x):
        """Computes class probabilities given the input.

        Args:
            x: input

        Returns:
            a vector representing P[y|x]
        """
        p_xd = self.kernel(x, self.x_a)  # (B, N)

        pyl, pyu = self.parameter_values()
        p_dy = torch.cat([pyl, pyu], dim=0)  # (N, K)

        p_xy = p_xd @ p_dy  # (B, K)
        p_x = p_xd.sum(-1)[:, None]  # (B, 1)
        p_y = p_xy / p_x  # (B, K)

        return p_y


def normal_perturb(x, std=1.):
    return torch.normal(x, std)


def uniform_perturb(x, std=1.):
    l = std * 12 ** 0.5
    return x + torch.rand(x.shape) * l - l / 2


def plot_predictions(model: SemisupKernelClassifier, x_l=None, y_l=None, x_u=None, title=None,
                     legend=False, other=dict()):
    """Plots predictions and correct labels

    Args:
        model (torch.nn.Module): A function that returns class probabilities for
        x_l (torch.Tensor): A batch of inputs (floats).
        y_l (torch.Tensor): A batch of labels (integers).
        x_u (torch.Tensor): A batch of unlabeled inputs (floats).
    """
    x_l = model.x_l if x_l is None else x_l
    y_l = model.x_l if y_l is None else y_l
    x_u = model.x_u if x_u is None else x_u
    x = torch.cat([x_l, x_u], dim=0)
    start, end = x.min(dim=0).values, x.max(dim=0).values
    plus = (end - start) * 0.2
    start, end = (start - plus, end + plus)

    # x_u = torch.cat([x_u] + [normal_perturb(x_u, std=0.2) for _ in range(5)])

    with torch.no_grad():
        x_g = torch.linspace(start.squeeze(-1), end.squeeze(-1), steps=400)[:, None]
        py_g = model(x_g)  # (N, C)
    num_classes = py_g.shape[1]

    # Plot
    if callable(get_figsize):
        w, h = get_figsize(0.6)
        fig, ax = plt.subplots(figsize=(w, h * 0.8))

    # Plot the density
    density = model.density(x_g, support=x)
    density /= density.max() * 2
    ax.fill(x_g.cpu().numpy(), density.cpu().numpy(),
            label=0 * r"$\propto$ density" + 1 * rf"$\propto\mathrm{{p}}(x)$",
            alpha=0.1, color='k', linewidth=0.0)

    # Plot the examples
    ax.vlines(x_l, ymin=0, ymax=1, color='C1', alpha=0.3, zorder=0)
    ax.vlines(x_u, ymin=0, ymax=1, color='C1', alpha=0.3, linestyle='-', label='Input', zorder=0)

    # Plot horizontal axes
    ax.axhline(y=0.5, color='k', alpha=0.3, zorder=0)
    # ax.axhline(y=0, color='k', alpha=0.3, zorder=0)

    # Plot the labeled examples
    if len(model.x_l) != len(x_l):
        pyl = F.one_hot(y_l.to(torch.long), model.s_u.shape[1])
        for i in range(0, num_classes) if num_classes > 2 else [int(num_classes == 2)]:
            pyl_c = pyl[:, i].cpu().numpy()
            ax.scatter(x_l[:, 0].cpu().numpy(), pyl_c, marker='D', color='C1', label='Label',
                       zorder=3)

    # Plot the parameters
    pyl, pyu = model.parameter_values()
    pyl, pyu = pyl.detach(), pyu.detach()
    for i in range(0, num_classes) if num_classes > 2 else [int(num_classes == 2)]:
        if len(model.x_l) > 0:
            pyl_c = pyl[:, i].cpu().numpy()
            ax.scatter(model.x_l[:, 0].cpu().numpy(), pyl_c, marker='D', color='C1', label='Label',
                       zorder=2)
        pyu_c = pyu[:, i].cpu().numpy()
        ax.scatter(model.x_u[:, 0].cpu().numpy(), pyu_c, marker='.', color='C0', alpha=0.3,
                   label=r'$\sigma(\theta_k)$', zorder=1)

    # Plot the model's prediction
    for i in range(0, num_classes) if num_classes > 2 else [int(num_classes == 2)]:
        ax.plot(x_g.cpu().numpy(), py_g[:, i].cpu().numpy(), color='C0',
                label=0 * rf"Class prob." + 1 * rf"$p_{{\Theta}}(y{{=}}{i}|x)$", zorder=4)

    # Set the plot title and axis labels
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("$x$")
    plt.xlim([start, end])

    # Plot other
    for label, (x, y) in other.items():
        ax.scatter(x.detach().cpu().numpy(), y.detach().cpu().numpy(), marker='x', color='C5',
                   label=label,
                   zorder=5)

    if legend:
        ax.legend(loc='upper right', fontsize="small")

    return fig, ax


def kl(pred, target):
    return F.kl_div(pred.log_softmax(dim=-1), target.softmax(dim=-1), reduction='none').sum(-1)


def rkl(pred, target):
    return kl(target, pred)


def kl_lp(pred, target_probs):
    return F.kl_div(pred.log_softmax(dim=-1), target_probs, reduction='none').sum(-1)


def js(a, b):
    m = (a.softmax(dim=-1) + b.softmax(dim=-1)) / 2
    return (kl_lp(a, m) + kl_lp(b, m)) / 2


def sqr(a, b):
    return (a - b).pow(2).mean(-1)


def train(model, x_l, y_l, x_u, num_iterations, lr=0.1, cons_loss_f=js, detach_clean=True,
          detach_pert=False, pert=normal_perturb, iter_end_callback=lambda vars: None,
          init_callback=lambda vars: None, hybrid=False):
    x_uns = torch.cat([x_l, x_u], dim=0)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0, betas=(0, 0), lr=lr)
    # optimizer = torch.optim.RMSprop(model.parameters(), weight_decay=0, alpha=0, lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), weight_decay=0, lr=lr * 20000)

    i = -1
    loss = torch.zeros(1)
    loss_c = torch.zeros(len(x_uns))

    for i in range(num_iterations):
        # Forward pass on labeled data
        p_y = model(x_l)
        loss_labeled = F.cross_entropy(p_y.log(), y_l)

        # Forward pass on unlabeled data
        p_y_uns = model(x_uns)

        E = 50
        loss_cons = 0
        for _ in range(E):
            x_p = pert(x_uns)
            p_y_p = model(x_p)

            if not hybrid and detach_clean or hybrid and i < num_iterations / 6:  # clean teacher
                p_y_uns = p_y_uns.detach()
            if not hybrid and detach_pert or hybrid and i >= num_iterations / 6:  # clean student
                p_y_p = p_y_p.detach()
            loss_cons = loss_cons + cons_loss_f(p_y_uns, p_y_p)

            # entropy = (-p_y * torch.log(p_y + 1e-8)).sum(dim=-1).mean()
        loss_cons /= E
        loss = loss_labeled + loss_cons.mean()

        if i == 0:
            init_callback({**locals(), 'i': -1})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_end_callback(locals())


def training_callback(state, x_l=None, y_l=None, x_u=None, legend=False):
    from argparse import Namespace
    st = Namespace(**state)

    step = 1
    if st.i % step == step - 1:
        print(f"iteration {st.i}: loss = {st.loss.item()}")

        # if st.i % (st.num_iterations // 10) == 0:
        plt.clf()
        fig, ax = plot_predictions(
            st.model, x_l=x_l, y_l=y_l, x_u=x_u,
            title=f"iter={st.i + 1}, loss_sup={st.loss_labeled.item():.3f}, loss_cons={st.loss_cons.mean().item():.3f}",
            legend=legend)
        # other=dict(loss_cons_10=(st.x_uns, st.loss_cons * 10)))
        path = Path(f"D:/figures/semisup/plot{st.i // step + 1}.png")
        path.parent.mkdir(exist_ok=True)
        fig.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close()
        # plt.show()
    # time.sleep(0.01)


def run(cons_loss_f, detach_clean, detach_pert, hybrid=False, legend=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x_l, y_l, x_u = [torch.tensor(a, device=device) for a in TriClusterDataset.get_data()]
    y_l = y_l.to(torch.int64)
    std = 1.
    std_pert = 1

    # model = SemisupKernelClassifier(x_l=x_l, y_l=y_l, x_u=x_u, num_classes=None,
    #                                 kernel=partial(normal_kernel, std=std)).to(device)

    model = SemisupKernelClassifier(x_l=x_l[:0], y_l=y_l[:0],
                                    x_u=torch.tensor(TriClusterDataset.get_uniform_data(),
                                                     device=device),
                                    num_classes=None,
                                    kernel=partial(normal_kernel, std=std)).to(device)

    plot_predictions(model, x_l=x_l, y_l=y_l, x_u=x_u, legend=legend)
    plt.show()

    callback = partial(training_callback, x_l=x_l, y_l=y_l, x_u=x_u, legend=legend)

    train(model, x_l, y_l, x_u, num_iterations=600, lr=0.02, cons_loss_f=cons_loss_f,
          detach_clean=detach_clean, detach_pert=detach_pert, hybrid=hybrid,
          pert=partial(uniform_perturb, std=std_pert), iter_end_callback=callback,
          init_callback=callback)

    plot_predictions(model, x_l=x_l, y_l=y_l, x_u=x_u, legend=legend)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cons_loss_f", type=str, default="js")
    parser.add_argument("--detach_clean", action='store_true')
    parser.add_argument("--detach_pert", action='store_true')
    parser.add_argument("--hybrid", action='store_true')
    parser.add_argument("--legend", action='store_true')
    args = parser.parse_args()

    run(cons_loss_f=eval(args.cons_loss_f), detach_pert=args.detach_pert,
        detach_clean=args.detach_clean, hybrid=args.hybrid, legend=args.legend)
