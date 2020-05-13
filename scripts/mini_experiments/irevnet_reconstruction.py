from torchvision.transforms.functional import to_tensor, to_pil_image
import torch
from PIL import Image
import matplotlib.pyplot as plt

import _context
from vidlu import factories, problem
import vidlu.modules as vm
import vidlu.modules.functional as vmf

img = Image.open('image.jpg')
images = [img]
x = to_tensor(img)[:, :80, :112].unsqueeze(0)

for i in range(4):
    model = factories.get_model(
        "IRevNet,backbone_f=t(init_stride=1,base_width=8,group_lengths=(8,8,8))",
        problem=problem.Classification(class_count=8),
        init_input=x)
    model_injective = vm.deep_split(model, 'backbone.concat')[0]

    h = model_injective(x)
    x_r = model_injective.inverse(h).squeeze().clamp(0, 1)

    img_r = to_pil_image(x_r)

    images += [img_r]

fig, axs = plt.subplots(1, len(images))
for i, im in enumerate(images):
    axs[i].imshow(im)
plt.show()

