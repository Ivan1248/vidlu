# based on Marin's script

import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
import os
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--div_fn", type=str, default="kl")
parser.add_argument("--xlabel", type=str, default=" ")
parser.add_argument("--ylabel", type=str, default=" ")
parser.add_argument("--semi", action='store_true')
parser.add_argument("--detach_clean", action='store_true')
parser.add_argument("--detach_pert", action='store_true')
parser.add_argument("--epochs", type=int, default=20000)
parser.add_argument("--pert", type=float, default=2)  # 29 31 32 33
parser.add_argument("--seed", type=int, default=33)  # 29 31 32 33
args = parser.parse_args()

sns.set(font_scale=2.5)
sns.set(font_scale=3, rc={'text.usetex': True})
plt.rcParams['text.latex.preamble'] = [r'\usepackage{bm}']

num_samples = 200
n_labeled = 3
X, Y = tuple(map(torch.from_numpy,
                 make_moons(n_samples=num_samples, noise=8e-2, random_state=args.seed,
                            shuffle=True)))
X = X.float()
n_val = int(num_samples * .8)
ignore_index = 2

data_train = {
    'X': X[:, 0][:int(num_samples * .8)],
    'Y': X[:, 1][:int(num_samples * .8)],
    'data': X[:int(num_samples * .8)],
    'class': Y[:int(num_samples * .8)],
}
data_train['class'][n_labeled:-n_labeled] = ignore_index

_, indices = data_train['class'].sort(descending=True)
for k in ['X', 'Y', 'data', 'class']:
    data_train[k] = data_train[k][indices]
data_train['size'] = [80, 80, 20]
data_train['sizes'] = torch.tensor(data_train['size'])[data_train['class']]

print(data_train['class'][data_train['class'].ne(ignore_index)])

data_val = {
    'X': X[:, 0][int(num_samples * .8):],
    'Y': X[:, 1][int(num_samples * .8):],
    'data': X[int(num_samples * .8):],
    'class': Y[int(num_samples * .8):],
    'size': [500, 500]
}
data_val['sizes'] = torch.tensor(data_val['size'])[data_val['class']]

device = torch.device('cpu')

epochs = args.epochs
detach_clean = args.detach_clean
detach_pert = args.detach_pert
has_semisup = True  # not (detach_clean and detach_pert)

noise = torch.distributions.normal.Normal(0, args.pert * 10e-2)
x_train, y_train = data_train['data'].to(device), data_train['class'].to(device)


def kl(x, y):
    return F.kl_div(x.log_softmax(dim=1), y.softmax(dim=1), reduce='batchmean')


def rkl(x, y):
    return kl(y, x)


def ce(x, y):
    return -y.softmax(dim=1).mul(x.log_softmax(dim=1)).sum(-1).mean()


def ent(x, y):
    return -x.softmax(dim=1).mul(x.log_softmax(dim=1)).sum(-1).mean()


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


model = nn.Sequential(
    nn.Linear(2, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 2)).to(device)
seed = 5
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
model.apply(init_weights)

criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

div_fn = eval(args.div_fn)
for i in range(epochs):
    # if i % (epochs / 20) == 0:
    # generate_plot(data_train, None)
    # plt.savefig(f'/tmp/toy_uda{"_on" if has_semisup else "_off"}_{i:06d}.png', dpi=100, facecolor='w', edgecolor='w',
    #             orientation='landscape', papertype=None, format='png',
    #             transparent=True, bbox_inches='tight', pad_inches=0.)
    out_c = model(x_train)
    loss_sup = criterion(out_c, y_train)
    if has_semisup:
        with torch.no_grad():
            x_train_p = x_train + noise.sample(x_train.shape).to(device)
        out_p = model(x_train_p)
        out_c = out_c.detach() if detach_clean else out_c
        out_p = out_p.detach() if detach_pert else out_p
        loss_un = div_fn(out_p, out_c)
        loss = loss_sup + loss_un # + 1e-2*sum(p.pow(2).sum() for p in model.parameters() if len(p.shape)>1)
    # lr = (epochs - i) / args.epochs
    # (loss * lr).backward()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if i % (epochs // 10) == 0:
        print(f'Loss: {loss.item():.3f} sup: {loss_sup.item():.3f}' + (
            f' uns:{loss_un.item():.3f}' if has_semisup else ''))

converged = loss_un.item() < 0.3 or args.detach_clean and args.detach_pert
device = torch.device('cpu')
model.to(device)

# generate_plot(data_train, 'a)' if not has_semisup else 'b)')
# title = f'semi={has_semisup} detach_clean={detach_clean} detach_pert={detach_pert} div_fn={args.div_fn} epochs={epochs}'

translate = dict()

title = " ".join([k if isinstance(v, bool) else f"{v}" for k, v in args.__dict__.items()
                  if v != False and k not in ["xlabel", "ylabel", "seed"]])
print(title)


# generate_plot(data_train, os.environ.get('TITLE'))

def generate_plot(model, data, title):
    markers = {0: "o", 1: "o", ignore_index: 'o'}
    xmin, xmax, ymin, ymax = data['X'].min(), data['X'].max(), data['Y'].min(), data['Y'].max()
    w, h = xmax - xmin, ymax - ymin
    xmin -= w * .1
    xmax += w * .1
    ymin -= h * .1
    ymax += h * .1
    with torch.no_grad():
        steps = 500
        lx = torch.linspace(xmin, xmax, steps=steps)
        ly = torch.linspace(ymin, ymax, steps=steps)
        xx0, xx1 = torch.meshgrid(lx, ly)
        grid = torch.stack((xx0.flatten(), xx1.flatten()), dim=1)
        values = model(grid.to(device)).softmax(dim=1)[..., 0].reshape((steps, steps)).data.to(
            'cpu')
    cmap_bg = sns.diverging_palette(220, 16, n=9, l=70, as_cmap=True, center='light')
    cmap = sns.color_palette(["#aa2211", "#1177aa", "#ffffff"])
    plt.contourf(xx0, xx1, values, cmap=cmap_bg)
    ax = sns.scatterplot(x="X", y="Y", hue="class", data=data, palette=cmap, legend=False,
                         # s=200,
                         sizes=(120, 500),
                         size="sizes",
                         markers=markers,
                         style="class",
                         edgecolor='#000000',
                         linewidth=0.2)
    if title is not None:
        ax.set_title(title)
    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    sns.despine()


generate_plot(model, data_train, title=None)

plt.show()

print(f"{converged=}{loss.item()=}")
print(args.xlabel, args.ylabel)

path = Path(f'd:/dump/semi_toy_{title.replace(" ", "_")}.pdf')
path.parent.mkdir(exist_ok=True)

plt.savefig(path, bbox_inches='tight', pad_inches=0)
