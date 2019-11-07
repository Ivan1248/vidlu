# list aprameters with shapes
print('\n'.join(f"{k} {tuple(v.shape)}" for k, v in trainer.model.state_dict().items()))

# semseg
visualization.view_predictions(
    data.train.map(lambda r, trainer=trainer: (
        trainer.prepare_batch((r.x.reshape((1,) + r.x.shape), r.y.reshape((1,) + r.y.shape)))[
            0].squeeze().permute(1, 2, 0).detach().cpu().numpy(), r.y.cpu().numpy())),
    infer=lambda x, trainer=trainer: trainer.model(
        torch.tensor(x).to(device=trainer.model.device).permute(2, 0, 1).unsqueeze(0)).argmax(
        1).squeeze().int().cpu().numpy())

visualization.view_predictions(
    data.train.map(lambda r, trainer=trainer: (
        (r.x.permute(1, 2, 0).detach().cpu().numpy(), r.y.cpu().numpy()))),
    infer=lambda x, trainer=trainer: trainer.model(
        torch.tensor(x).to(device=trainer.model.device).permute(2, 0, 1).unsqueeze(0)).argmax(
        1).squeeze().int().cpu().numpy())

# semseg, adversarial
visualization.view_predictions(
    data.train.map(lambda r, trainer=trainer: (
        trainer.attack.perturb(trainer.model,
                               *trainer.prepare_batch((r.x.unsqueeze(0), r.y.unsqueeze(0))))
            .squeeze().permute(1, 2, 0).detach().cpu().numpy(),
        r.y.cpu().numpy())),
    infer=lambda x, trainer=trainer: trainer.model(
        torch.tensor(x).to(device=trainer.model.device).permute(2, 0, 1).unsqueeze(0)).argmax(
        1).squeeze().int().cpu().numpy())

# semseg, VAT
visualization.view_predictions(
    data.train.map(lambda r, trainer=trainer: (
        trainer.attack.perturb(trainer.model,
                               trainer.prepare_batch((r.x.unsqueeze(0), r.y.unsqueeze(0)))[0])
            .squeeze().permute(1, 2, 0).detach().cpu().numpy(),
        r.y.cpu().numpy())),
    infer=lambda x, trainer=trainer: trainer.model(
        torch.tensor(x).to(device=trainer.model.device).permute(2, 0, 1).unsqueeze(0)).argmax(
        1).squeeze().int().cpu().numpy())

# save and view images
x, x_av = state.output.x, state.output.x_adv
# columns = 4  # semseg
columns = 16  # cifar
from torchvision import utils
import os

path = "/tmp/images/debug.png"
utils.save_image(
    utils.make_grid(torch.stack([x, x_av], 1).view(), columns), path)
os.system('DISPLAY=localhost:10.0 geeqie /tmp/images/debug.png')

# print confusion matrix
np.set_printoptions(edgeitems=30, linewidth=100000)
print(repr(
    np.array(next(m for m in trainer.metrics if type(m).__name__.startswith('Classification')).cm,
             dtype=np.int64)))

# print metrics
print(trainer.metrics['ClassificationMetrics'].compute())

# hooks
trainer.model.backbone.bulk.unit0_0.fork.block.norm1.orig.weight
trainer.model.backbone.bulk.unit0_0.fork.block.norm1.orig.weight.grad
pp = {n: p for n, p in trainer.model.named_parameters() if torch.all(p == 0)}
print('\n'.join(pp.keys()))

trainer.model.backbone.bulk.unit0_0.fork.block.norm1.register_forward_hook(
    lambda s, inp, out, trainer=trainer: print(
        trainer.model.backbone.bulk.unit0_0.fork.block.norm1.orig.weight))
trainer.model.backbone.bulk.unit0_0.fork.block.norm1.register_forward_hook(
    lambda s, inp, out, trainer=trainer: print(
        trainer.model.backbone.bulk.unit0_0.fork.block.norm1.orig.bias))

trainer.model.backbone.bulk.unit0_0.fork.block.norm1.register_forward_hook(
    lambda s, inp, out: print(inp[0].abs().max()))
trainer.model.backbone.bulk.unit0_0.fork.block.norm1.register_forward_hook(
    lambda s, inp, out: print(inp[0].grad))
trainer.model.backbone.bulk.unit0_0.fork.block.norm1.register_forward_hook(
    lambda s, inp, out: print(inp[0].grad, out.grad))
trainer.model.backbone.bulk.unit0_0.fork.block.norm1.register_forward_hook(
    lambda s, inp, out: print(out))
trainer.model.backbone.bulk.unit0_0.fork.block.norm1.register_backward_hook(
    lambda s, ig, og: print(ig[0].grad, og))
trainer.model.backbone.bulk.unit0_0.fork.block.norm1.register_backward_hook(
    lambda s, ig, og: print(og))
trainer.model.backbone.bulk.unit0_0.fork.block.act1.register_backward_hook(
    lambda s, ig, og: print(ig[0].abs().sum()))
trainer.model.backbone.bulk.unit0_0.fork.block.act1.register_forward_hook(
    lambda s, inp, out: print(inp[0].abs().max()))
trainer.model.backbone.bulk.unit0_0.fork.block.act1.register_forward_hook(
    lambda s, inp, out: print((inp[0] > 0).float().mean()))

# print calls

vidlu.utils.debug.tracecalls()

# make eval attack stronger
trainer.eval_attack.stop_on_success = False
trainer.eval_attack.step_count = 50
trainer.eval_attack.eps *= 4

# make eval attack stronger and opposite loss sign
trainer.eval_attack.stop_on_success = False
trainer.eval_attack.step_count = 50
trainer.eval_attack.eps *= 4
trainer.eval_attack.loss = lambda *a, **k: -trainer.eval_attack.loss(*a, **k)

# show adversarial examples
import torch

with torch.no_grad():
    from torchvision.utils import make_grid


    def show(img):
        import numpy as np
        import matplotlib.pyplot as plt
        npimg = img.detach().cpu().numpy()
        plt.close()
        plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')


    N = 16
    adv = state.output.x_adv[:N]
    clean = state.output.x[:N]
    diff = 0.5 + (adv - clean) * 255 / 80
    pred = state.output.other_outputs_adv.hard_prediction
    target = state.output.target

    fooled = (pred != target)[:N]
    fooled = fooled.reshape(-1, *[1] * (len(adv.shape) - 1))
    fooled = fooled.float() * (adv * 0 + 1)

    class_repr = [None] * 10
    for i, c in enumerate(target):
        if class_repr[c] is None:
            class_repr[c] = state.output.x[i]
    for i, x in enumerate(class_repr):
        if x is None:
            x = 0 * adv[0]

    predicted_class_representatives = list(map(class_repr.__getitem__, pred[:N]))

show(make_grid(
    sum((list(x) for x in [clean, adv, diff, fooled, predicted_class_representatives]), []),
    nrow=len(adv)))

# tent hyperparameters WRN
from torch import optim

delta_params = [v for k, v in trainer.model.named_parameters() if k.endswith('delta')]
other_params = [v for k, v in trainer.model.named_parameters() if not k.endswith('delta')]
trainer.optimizer = optim.Adam(
    [dict(params=other_params), dict(params=delta_params, weight_decay=0.12)], lr=1e-3,
    weight_decay=1e-6)
for m in trainer.model.modules():
    if hasattr(m, 'max_delta'):
        print(m.max_delta, m.delta)
        m.max_delta = 1.0
        m.min_delta = 0.05

for m in trainer.model.modules():
    if 'ReLU' in str(type(m)):
        print(type(m))

# tent adversairal example visualization pre-code
state = torch.load(
    '/home/igrubisic/data/states/cifar10{trainval,test}/ResNetV2,backbone_f=t(depth=18,small_input=True,block_f=t(act_f=mc.Tent))/AdversarialTrainer,++{++configs.wrn_cifar_tent,++configs.adversarial},attack_f=attacks.DummyAttack,eval_attack_f=partial(configs.madry_cifar10_attack,step_count=7,stop_on_success=True)/_/91/state.pth')
trainer.model.load_state_dict(state['model'])
from vidlu.training.configs import *

trainer.eval_attack = madry_cifar10_attack(trainer.model, step_count=50, eps=40 / 255)

"""
from torch import optim
delta_params = [v for k, v in trainer.model.named_parameters() if k.endswith('delta')]
other_params = [v for k, v in trainer.model.named_parameters() if not k.endswith('delta')]
trainer.optimizer=optim.SGD([dict(params=other_params), dict(params=delta_params, weight_decay=4e-2)], lr=1e-3, momentum=0.9, weight_decay=5e-4)"""

# activations
from vidlu.modules import with_intermediate_outputs

for i in range(4):
    print((with_intermediate_outputs(trainer.model, [f'backbone.act{i}_1'])(state.output.x)[1][
               0] != 0).float().mean())

from vidlu.modules import with_intermediate_outputs

for i in range(4):
    print((with_intermediate_outputs(trainer.model, [f'backbone.norm{i}_1'])(state.output.x)[1][
        0]).float())

from vidlu.modules import with_intermediate_outputs

for i in range(4):
    print((with_intermediate_outputs(trainer.model, [f'backbone.norm{i}_1'])(state.output.x)[1][
               0] > 0.5).float().mean())

from vidlu.modules import with_intermediate_outputs

for k, v in trainer.model.named_buffers():
    if 'bias' in k:
        print(v)
