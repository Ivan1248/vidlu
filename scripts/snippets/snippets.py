# list aprameters with shapes
from vidlu.utils.presentation import visualization

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

trainer.attack.minimize = False
trainer.attack.eps = 3 / 255
trainer.attack.step_size = 2 / 255
trainer.attack.step_count = 40

visualization.view_predictions(
    data.test.map(lambda r, trainer=trainer: (
        trainer.attack.perturb(trainer.model,
                               *trainer.prepare_batch((r.x.unsqueeze(0), r.y.unsqueeze(0))))
            .squeeze().permute(1, 2, 0).detach().cpu().numpy(),
        r.y.cpu().numpy())),
    infer=lambda x, trainer=trainer: trainer.model(
        torch.tensor(x).to(device=trainer.model.device).permute(2, 0, 1).unsqueeze(0)).argmax(
        1).squeeze().int().cpu().numpy(),
    save_dir="/home/igrubisic/robust_seg_eps3_it40")

# semseg, adversarial - shifted labels, targeted

ds = data.train


def shift_label(i, ds=ds):
    from vidlu.data import Record
    # yt = ds[(i + 1) % len(ds)].y
    yt = ds[(i + 1) % len(ds)].y
    yt[yt != 13] = 0
    return Record(x=ds[i % len(ds)].x, y=ds[i % len(ds)].y, yt=yt)


def single_label(i, ds=ds):
    from vidlu.data import Record
    yt = ds[i % len(ds)].y.clone().fill_(13)
    return Record(x=ds[i % len(ds)].x, y=ds[i % len(ds)].y, yt=yt)


ds = type(ds).from_getitem_func(single_label, len(ds), info=ds.info)
trainer.attack.minimize = True
trainer.attack.eps = 100 / 255
trainer.attack.step_size = 2 / 255
trainer.attack.step_count = 150


def greyed_image_as(x):
    return x * 0.5  # REMOVEREMOVEREMOVEREMOVEREMOVEREMOVEREMOVEREMOVEREMOVEREMOVEREMOVE
    x = x * 0.5 + x.mean()
    x[0, 0, 0] = 0
    x[0, 0, 1] = 1
    return x


def process_example(r, trainer=trainer, empty_image_as=greyed_image_as):
    return (trainer.attack.perturb(trainer.model,
                                   *trainer.prepare_batch(
                                       (empty_image_as(r.x.unsqueeze(0)), r.yt.unsqueeze(0))))
            .squeeze().permute(1, 2, 0).detach().cpu().numpy(),
            r.yt.cpu().numpy())


visualization.view_predictions(
    ds.map(process_example),
    infer=lambda x, trainer=trainer: trainer.model(
        torch.tensor(x).to(device=trainer.model.device).permute(2, 0, 1).unsqueeze(0)).argmax(
        1).squeeze().int().cpu().numpy(),
    save_dir="/home/igrubisic/robust_seg_2")

# semseg, adversarial, PGD iterations

# python run.py train "cityscapes{train,val}" id "SwiftNet,backbone_f=t(depth=18,small_input=False)" "tc.swiftnet_cityscapes,tc.adversarial,epoch_count=200,attack_f=partial(tc.madry_cifar10_attack,step_count=7,eps=2/255,step_size=0.5/255),eval_attack_f=t(step_count=10),eval_batch_size=1" --params "id:/home/igrubisic/data/states/cityscapes{train,val}/SwiftNet,backbone_f=t(depth=18,small_input=False)/tc.swiftnet_cityscapes,tc.adversarial,epoch_count=200,attack_f=partial(tc.madry_cifar10_attack,step_count=7,eps=2/255,step_size=0.5/255),eval_attack_f=t(step_count=10),eval_batch_size=4/resnet(backbone),backbone.backbone+resnet18-5c106cde.pth/_/200/model_state.pth"
# ffmpeg -i %05d.png -vcodec libx264 -crf 2 -filter:v scale=1024:-1 robust_pgd_50_3_200.avi
trainer.attack.minimize = False
trainer.attack.eps = 20 / 255
trainer.attack.step_size = 1 / 255
trainer.attack.step_count = 200
trainer.attack.rand_init = False

from vidlu.modules.components import GaussianFilter2D

low_pass = GaussianFilter2D(5)


def reg(x, delta, low_pass=low_pass):
    return 1e15 / (x.shape[2] * x.shape[3]) * (delta - low_pass(delta)).pow_(2).sum((2, 3))


def reggrad(x, delta, y, t):
    return 1e12 * ((delta[:, :, :-1, :] - delta[:, :, 1:, :]).pow_(2).mean((2, 3))
                   + (delta[:, :, :, :-1] - delta[:, :, :, 1:]).pow_(2).mean((2, 3)))


def segreggrad(x, delta, y, t):
    mask = ((t[:, :-1, :-1] == t[:, 1:, :-1]) | (t[:, :-1, :-1] == t[:, :-1, 1:])
            ).view(1, -1, t.shape[1] - 1, t.shape[2] - 1)
    return 1e12 * ((delta[:, :, :-1, :-1] - delta[:, :, 1:, :-1]).pow_(2)
                   + (delta[:, :, :-1, :-1] - delta[:, :, :-1, 1:]).pow_(2)).mul_(mask).mean((2, 3))


def ent(y, t, attack=trainer.attack):
    from vidlu.modules import losses
    loss = losses.entropy_l(y)  # .mean()
    print(y.mean())
    return loss


def logit_sum(y, t, attack=trainer.attack):
    loss = y.mean(1)
    print(y.mean())
    return -loss


# trainer.attack.loss = ent #, segreggrad
# trainer.attack.loss = NLLLossWithLogits()  # , segreggrad
trainer.attack.loss = logit_sum  # , segreggrad

visualization.generate_adv_iter_segmentations(dataset=data.test.map(trainer.prepare_batch),
                                              model=trainer.model,
                                              attack=trainer.attack,
                                              save_dir="/home/igrubisic/logits_min")

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
x, x_av = state.output.x, state.output.x_p
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

def show_adversarial_examples(trainer, state, **kwargs):
    import vidlu.modules.inputwise as vmi

    # trainer.eval_attack.pert_model_f = vmi.Warp
    #
    # trainer.eval_attack.eps = 0.2
    # trainer.eval_attack.step_size = 0.01
    trainer.eval_attack.step_count = 1000
    # trainer.eval_attack.stop_on_success = True

    import torch

    with torch.no_grad():
        from torchvision.utils import make_grid

        def show(img):
            import numpy as np
            import matplotlib.pyplot as plt
            npimg = img.detach().cpu().numpy()
            plt.close()
            plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
            plt.show()

        N = 16
        #adv = trainer.eval_attack.perturb(trainer.model, state.output.x, state.output.target)
        adv = state.output.x_p[:N]
        clean = state.output.x[:N]
        diff = 0.5 + (adv - clean) * 255 / 80
        pred = state.output.other_outputs_p.hard_prediction[:len(state.output.target)]
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


show_adversarial_examples(**locals())


# activations

def activations(trainer, **kwargs):
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

    for k, v in trainer.model.named_buffers():
        if 'bias' in k:
            print(v)


# i-RevNet reconstruction


def visualize_reconstructions(trainer, **kwargs):
    def random_tps_warp(x):
        import vidlu.modules.functional as vmf
        import numpy as np
        c_src = vmf.uniform_grid_2d((2, 2)).view(-1, 2).unsqueeze(0).to(x.device)
        offsets = c_src * 0
        offsets[..., 0, 0] = np.random.uniform(0, 0.5)
        offsets[..., 0, 1] = np.random.uniform(0, 0.5)
        from vidlu.modules.inputwise import TPSWarp
        warp = TPSWarp(forward=False, control_grid_shape=(2, 2))
        warp(x)
        with torch.no_grad():
            warp.offsets.add_(offsets)
        return warp

    from torchvision.transforms.functional import to_tensor, to_pil_image
    import torch
    from PIL import Image
    import matplotlib.pyplot as plt

    import vidlu.modules as vm

    img = Image.open('mini_experiments/image.jpg')
    x1 = to_tensor(img)[:, :80, :112].unsqueeze(0).to(trainer.model.device)  # 1CHW

    warp = random_tps_warp(x1)
    xs = [x1, torch.flip(x1, [1, 2]), warp(x1)]

    with torch.no_grad():
        try:
            model_injective = vm.deep_split(trainer.model, 'backbone.concat')[0]
        except TypeError as e:
            model_injective = vm.deep_split(trainer.model.backbone.backbone, 'concat')[0]
        hs = list(map(model_injective, xs))
        hs.append((hs[0] + hs[1]) / 2)
        hs.append(warp(hs[0]))
        hs.append(torch.flip(hs[0], [2]))
        # hs.append(torch.randn_like(hs[0]))

        x_rs = [model_injective.inverse(h).clamp(0, 1) for h in hs]
        x_rs = [model_injective.inverse(torch.roll(h, [0, 0], dims=[2, 3])).clamp(0, 1) for h in hs]

    images = [to_pil_image(x[0].cpu()) for x in xs]
    images += [to_pil_image(x_r[0].cpu()) for x_r in x_rs]

    for x in x_rs:
        print(x.shape, x.dtype)

    w = int(len(images) ** 0.5 + 0.5)
    h = len(images) // w

    fig, axs = plt.subplots(h, w)
    axs = axs.flat
    for i, im in enumerate(images):
        axs[i].imshow(im)
    plt.show()


visualize_reconstructions(**globals(), **locals())


def visualize_inverse_adv_examples(trainer, state, **kwargs):
    from torchvision.transforms.functional import to_tensor, to_pil_image
    import torch
    import matplotlib.pyplot as plt
    from itertools import chain
    import vidlu.utils.presentation.visualization

    n = 6

    xs = state.batch[0].to('cuda')
    # logits, *other_out = state.output.other_outputs.full_output
    # breakpoint()
    logits, *other_out = trainer.model(xs)

    with torch.no_grad():
        trainer.model.eval()
        pred = logits[:, :10].argmax(1)
        print("c", logits[:10, -1], pred)
        logits_m = logits.roll(1, dims=0)
        pred_m = logits_m[:, :10].argmax(1)
        print("a", logits_m[:10, -1], (pred_m == pred).float().mean())
        # logits_m[:, -1].neg_()
        print(logits_m[:, -1].roll(1, dims=0).sigmoid().mean())
        for i in range(3):
            xs_adv = trainer.model.inverse((logits_m, *other_out))  # .clamp(-1, 1)
            logits_r, *other_out_r = trainer.model(xs_adv)
            pred_r = logits_r[:, :10].argmax(1)
            print("r", logits_r[:10, -1].sigmoid().mean(), (pred_r == pred).float().mean())
            xs_adv2 = trainer.model.inverse((logits_r, *other_out))  # .clamp(-1, 1)
            print(xs_adv2.min(), xs_adv2.max())
            logits_m = logits_r
            # xs_adv2 = trainer.model.inverse((logits_r, *other_out_r)).clamp(-1,1)

    xs = xs[:n]
    xs_adv = xs_adv[:n]
    deltas = xs_adv - xs + 0.5

    images = [to_pil_image(x.cpu().clamp(-0.999, 0.999)) for x in chain(xs, xs_adv, deltas, xs_adv2[:n])]

    h = 4  # len(images) // w
    w = len(images) // h  # int(len(images) ** 0.5 + 0.5)

    fig, axs = plt.subplots(h, w)
    axs = axs.flat
    for i, im in enumerate(images):
        axs[i].imshow(im)
        axs[i].set_axis_off()
    plt.show()


visualize_inverse_adv_examples(**{**globals(), **locals()})


# i-RevNet interpolation


def visualize_potty_interpolation(trainer, state, **kwargs):
    from torchvision.transforms.functional import to_tensor, to_pil_image
    import torch
    import matplotlib.pyplot as plt
    import vidlu.modules as vm
    import PIL.Image as pimg
    from torch.nn.functional import interpolate

    papiga = to_tensor(pimg.open("/home/igrubisic/Potty3.webp"))

    x = state.output.x[:20].clone()
    x[0, :, 10:250, 20:230] = papiga

    with torch.no_grad():
        # model_inj = vm.deep_split(trainer.model, 'backbone.concat')[0]
        # model_inj = trainer.model.backbone.backbone
        model_inj = vm.deep_split(trainer.model.backbone.backbone, 'concat')[0]
        h = model_inj(x)
        hr = torch.roll(h, [1], dims=[0])
        h2 = h.clone()
        h_papiga = interpolate(hr, scale_factor=1, mode="bilinear")
        h2[:, :, 2:10, 2:12] = h_papiga[:, :, 2:10, 2:12]
        hr = h2
        xint = [model_inj.inverse((1 - a) * h + a * hr) for a in np.linspace(0, 1, 5)]
        # xint = [model_inj.inverse(torch.randn_like(h)) for a in np.linspace(0, 1, 5)]

    imageses = [[to_pil_image(x.clamp(0, 0.99).cpu()) for x in xs] for xs in xint]

    fig, axeses = plt.subplots(len(imageses), len(imageses[0]))
    for axes, images in zip(axeses, imageses):
        for ax, im in zip(axes, images):
            ax.axis("off")
            ax.imshow(im)
    plt.subplots_adjust(wspace=.05, hspace=.05)
    plt.show()


visualize_potty_interpolation(**{**globals(), **locals()})


def visualize_latent_imagenet_pasting(trainer, state, **kwargs):
    from torchvision.transforms.functional import to_tensor, to_pil_image
    import torch
    import matplotlib.pyplot as plt
    import vidlu.modules as vm
    import PIL.Image as pimg
    from torch.nn.functional import interpolate
    import vidlu.transforms.image as vti

    pimages = ['/home/shared/datasets/ILSVRC2015/Data/CLS-LOC/train/n01440764/n01440764_10040.JPEG',
               '/home/shared/datasets/ILSVRC2015/Data/CLS-LOC/train/n01440764/n01440764_10048.JPEG',
               '/home/shared/datasets/ILSVRC2015/Data/CLS-LOC/train/n01440764/n01440764_10074.JPEG',
               '/home/shared/datasets/ILSVRC2015/Data/CLS-LOC/train/n01440764/n01440764_1009.JPEG',
               '/home/shared/datasets/ILSVRC2015/Data/CLS-LOC/train/n01440764/n01440764_10108.JPEG',
               '/home/shared/datasets/ILSVRC2015/Data/CLS-LOC/train/n01440764/n01440764_10251.JPEG',
               '/home/shared/datasets/ILSVRC2015/Data/CLS-LOC/train/n01440764/n01440764_10293.JPEG',
               '/home/shared/datasets/ILSVRC2015/Data/CLS-LOC/train/n01440764/n01440764_10323.JPEG',
               '/home/shared/datasets/ILSVRC2015/Data/CLS-LOC/train/n01440764/n01440764_10342.JPEG',
               '/home/shared/datasets/ILSVRC2015/Data/CLS-LOC/train/n01440764/n01440764_10361.JPEG']

    # LURD
    boxes = [(64, 103, 369, 277),
             (41, 142, 256, 271),
             (47, 241, 367, 368),
             (202, 105, 354, 182),
             (136, 360, 570, 542),
             (56, 230, 392, 362),
             (57, 127, 465, 281),
             (148, 167, 412, 270),
             (61, 85, 403, 196),
             (70, 221, 407, 367)]

    pimages = [to_tensor(pimg.open(x)) for x in pimages]

    pimages = [x[:, :x.shape[1] // 32 * 32, :x.shape[2] // 32 * 32] for x in pimages]

    # images = [x[:, l:l + (r - l) // 32 * 32, u:u + (u - d) // 32 * 32]
    #          for x, (l, u, r, d) in zip(images, boxes)]

    bimages = state.output.x[:len(pimages)].clone()
    if len(bimages) < len(pimages):
        bimages = torch.cat([bimages] * (len(pimages) // len(bimages) + 1), dim=0)

    with torch.no_grad():
        model_inj = vm.deep_split(trainer.model.backbone.backbone, 'concat')[0]
        zb = model_inj(bimages)
        zp = [model_inj(p.unsqueeze(0).to(zb.device)).squeeze(0) for p in pimages]
        zp = [p[:, u // 16:d // 16, l // 16:r // 16]
              for p, (l, u, r, d) in zip(zp, boxes)]
        for i, p in enumerate(zp):
            # _, h, w = p.shape
            # p = torch.nn.functional.adaptive_avg_pool2d(p.unsqueeze(0), (h // 2, w // 2)).squeeze(0)
            _, h, w = p.shape
            _, _, hb, wb = zb.shape
            print(zb.shape, p.shape, len(bimages))
            # zb[i, :, :h, :w] = p[:, :hb, :wb]
            zb[i, :, 2:h + 2, 3:w + 3] = p[:, :hb, :wb]
        xr = model_inj.inverse(zb)
        print(xr.min(), xr.max())

    imageses = [to_pil_image(x.clamp(0.01, 0.99).cpu()) for x in xr]
    imageses = [imageses[:3],
                imageses[3:6],
                imageses[6:9]]

    fig, axeses = plt.subplots(len(imageses), len(imageses[0]))
    for axes, pimages in zip(axeses, imageses):
        for ax, im in zip(axes, pimages):
            ax.axis("off")
            ax.imshow(im)
    plt.subplots_adjust(wspace=.05, hspace=.05)
    plt.show()


visualize_latent_imagenet_pasting(**{**globals(), **locals()})


def visualize_interpolation_seq(trainer, state, **kwargs):
    from torchvision.transforms.functional import to_tensor, to_pil_image
    import torch
    import matplotlib.pyplot as plt
    import vidlu.modules as vm
    import PIL.Image as pimg
    from torch.nn.functional import interpolate
    from pathlib import Path
    import numpy as np

    path = Path("/home/shared/datasets/Cityscapes/leftImg8bit_sequence/val/frankfurt")
    x = [to_tensor(pimg.open(im)) for im in sorted(list(path.iterdir()))[:4 * 2:2]]
    x = torch.stack(x)
    x = interpolate(x, size=state.output.x.shape[-2:]).to(state.output.x.device)

    with torch.no_grad():
        model_inj = vm.deep_split(trainer.model.backbone.backbone, 'concat')[0]
        h = model_inj(x)
        hr = torch.roll(h, [1], dims=[0])
        xint = [model_inj.inverse((1 - a) * h + a * hr) for a in np.linspace(0, 1, 3)]

    imageses = [[to_pil_image(x.clamp(0, 0.99).cpu()) for x in xs] for xs in xint]

    fig, axeses = plt.subplots(len(imageses), len(imageses[0]))
    for axes, images in zip(axeses, imageses):
        for ax, im in zip(axes, images):
            ax.axis("off")
            ax.imshow(im)
    plt.subplots_adjust(wspace=.05, hspace=.05)
    plt.show()


visualize_interpolation_seq(**{**globals(), **locals()})


# noisy batchnorm stats

def noisy_batchnorm_stats(trainer: "vidlu.training.Trainer", state, data, **kwargs):
    from torch import nn
    import math

    model = trainer.model

    metric_name = 'mIoU'
    # baseline = .9454
    # baseline = .7398
    baseline = .7539
    jitter = False

    perfs = []
    ns = [2 ** p for p in range(0, int(math.log2(len(data.train))) + 1)]
    ns.append(len(data.train))
    print(ns)
    for n in ns:
        bs = min(n, trainer.batch_size)

        data_train = trainer.simple_or_zip_data_loader(data.train.map(trainer.jitter) if jitter else data.train,
                                                       batch_size=bs, shuffle=False, drop_last=True)
        for m in [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]:
            m.momentum = None
            m.reset_running_stats()
        model.train()
        data_iter = iter(data_train)
        with torch.no_grad():
            for i in range(n // bs):
                x = trainer.prepare_batch(next(data_iter))[0]
                model(x)
        print(f"\nN = {n}, Batch size = {bs}")
        s = trainer.eval(data.test)
        perfs.append(s.metrics[metric_name])

    print(perfs)

    import matplotlib.pyplot as plt
    plt.plot(ns, [baseline for _ in ns], label='test_baseline')
    plt.plot(ns, perfs, label='test')
    plt.xscale("log")
    plt.xlabel("N")
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()


noisy_batchnorm_stats(**locals())


# batchnorm ensemble

def batchnorm_ensemble(trainer, state, data, **kwargs):
    import torch
    from torch import nn
    from vidlu.training.steps import ClassifierEnsembleEvalStep

    metric_name = 'A'
    baseline = .7539
    baseline = .9460
    jitter = False

    perfs = []
    ns = [2 ** p for p in range(0, 5 + 1)]
    print(ns)

    @torch.no_grad()
    def model_iter(model, n=3, batch_count=16):
        data_train = trainer.simple_or_zip_data_loader(data.train.map(trainer.jitter) if jitter else data.train,
                                                       batch_size=trainer.batch_size, shuffle=True, drop_last=True)
        for i in range(n):
            for m in [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]:
                m.momentum = None
                m.reset_running_stats()
            is_training = model.training
            model.train()
            data_iter = iter(data_train)
            for _ in range(batch_count):
                x = trainer.prepare_batch(next(data_iter))[0]
                model(x)
            model.train(is_training)
            yield model

    @torch.no_grad()
    def combine(xs):
        s, n = next(xs), 1
        for x in xs:
            s += x.softmax(dim=-1)
            n += 1
        s /= n
        return s

    for n in ns:
        trainer.eval_step = ClassifierEnsembleEvalStep(model_iter=partial(model_iter, n=n), combine=combine)
        print(f"\nN = {n}")
        with torch.no_grad():
            s = trainer.eval(data.test)
        perfs.append(s.metrics[metric_name])
        del s

    print(ns)
    print(perfs)

    import matplotlib.pyplot as plt
    plt.plot(ns, [baseline for _ in ns], label='test_baseline')
    plt.plot(ns, perfs, label='test')
    plt.xscale("log")
    plt.xlabel("N")
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()


batchnorm_ensemble(**locals())

# 8e-6
