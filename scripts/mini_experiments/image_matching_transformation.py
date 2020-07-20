from functools import partial

from PIL import Image
import matplotlib.pyplot as plt
import torch

# noinspection PyUnresolvedReferences
import _context
import dirs
import vidlu.training.robustness.attacks as vtaa
import vidlu.training.robustness.perturbation as vtap
import vidlu.transforms.image as vti
import vidlu.modules as vm
import vidlu.factories as vf
from vidlu.factories import problem

images = [Image.open(x) for x in reversed(['dz_dan.png', 'dz_noc.png'])]

images = list(map(vti.pil_to_torch, images))
images = [vti.hwc_to_chw(x).float().div(255).unsqueeze(0) for x in images]
if torch.cuda.is_available():
    images = [x.cuda() for x in images]

attack = vtaa.PertModelAttack(
    pert_model_f=partial(vtap.ChannelGammaHsv, forward_arg_count=1),
    loss=lambda a, b: -(a - b).abs(),
    step_count=40,
    step_size=0.05,
    projection=lambda x: x)

images += [images[-1]]

use_model = True
if use_model:
    model = vf.get_model("SwiftNet,backbone_f=t(depth=18,small_input=False)",
                         input_adapter_str="standardize(mean=[.485,.456,.406],std=[.229,.224,.225])",
                         problem=problem.SemanticSegmentation(y_shape=images[0].shape[-2:],
                                                              class_count=19),
                         init_input=images[0])
    params, submodule_path = vf.get_translated_parameters("swiftnet:swiftnet_ss_cs.pt",
                                                          params_dir=dirs.PRETRAINED)
    model.load_state_dict(params)
    model, _ = vm.deep_split(model.backbone.backbone, "bulk.unit0_0")
    #model, _ = vm.deep_split(model.backbone.backbone, "root.conv")
    #model = vm.Identity()
    model = model.cuda()
else:
    model = vm.Identity()

images[1] = attack.perturb(model, images[1], model(images[0]).detach(),
                           backward_callback=lambda s: print(s.step, s.loss_sum, s.pert_model))

images = [vti.torch_to_pil(vti.chw_to_hwc(x.squeeze().mul(255).byte().cpu().detach()))
          for x in images]

fig, axs = plt.subplots(len(images))
for i, im in enumerate(images):
    axs[i].imshow(im)
plt.show()
