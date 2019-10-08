from collections import defaultdict

from IPython import embed

import torchvision
import torch

from _context import vidlu, dirs

from vidlu import factories, models, problem, data, parameters, data_utils, modules

# data = data.datasets.HBlobs(example_shape=(32, 32, 3))

dataset = factories.get_data("TinyImagenet{val}", dirs.DATASETS)['TinyImagenet']['val']
dataset = factories.get_data_preparation(dataset)(dataset)
input = next(iter(data_utils.DataLoader(dataset, batch_size=1)))[0]

resnet_my = factories.get_model(
    model_str='ResNetV1,backbone_f=t(depth=18,small_input=False)',
    input_adapter_str='id', problem=problem.Classification(dataset.info.class_count),
    init_input=input)
resnet_my.eval()

resnet_tv = torchvision.models.resnet18(num_classes=dataset.info.class_count)
state = factories.get_translated_parameters("resnet", state_dict=resnet_tv.state_dict())
resnet_tv.eval()

resnet_my.load_state_dict(state)

layer_pairs = {
    'conv1':
        'backbone.root.conv',
    'bn1':
        'backbone.root.norm',
    'relu':
        'backbone.root.act',
    'maxpool':
        'backbone.root.pool',
    'layer1.0':
        'backbone.bulk.unit0_0',
    'layer4':
        'backbone.bulk',
    'avgpool':
        'head.pre_logits_mean',
    'fc':
        'head.logits',
}

output_pairs = defaultdict(list)

for tv, my in layer_pairs.items():
    def record_input(module, input, output, k=tv):
        output_pairs[k].append(output)


    modules.get_submodule(resnet_tv, tv).register_forward_hook(record_input)
    modules.get_submodule(resnet_my, my).register_forward_hook(record_input)

new_state = resnet_my.state_dict()
# resnet_my.load_state_dict(new_state)
# state = resnet_my.state_dict()

output_my = resnet_my(input)
for k, v in resnet_my.named_modules():
    print(k)
print()
for k, v in resnet_tv.named_modules():
    print(k)
print()
# output_my = resnet_my(input.detach().clone())
output_tv = resnet_tv(input)

"""try:
    output_my = resnet_my(input)
    output_tv = resnet_tv(input)
    assert torch.all(output_my == output_tv)
except AssertionError as ex:
    pass
"""

for k, (my, tv) in output_pairs.items():
    err = (my - tv).abs() / (torch.min(my.abs() + tv.abs()) + 1e-16)

    print(
        f"_{err.min().item():.1e} #{err.median().item():.1e} -{err.mean().item():.1e} ^{err.max().item():.1e}",
        k)
    # err.min().item(),, err.mean().item()
print('-\n-')

with torch.no_grad():
    for k in state.keys():
        my, tv = new_state[k], state[k]
        if len(my.shape) == len(tv.shape) == 0:
            print(f'[]=={k}')

        err = (my - tv).abs() / (torch.min(my.abs() + tv.abs()) + 1e-16)

        max_ = err.max().item()
        if max_ == 0:
            print('OK', k)
        else:
            print(f"_{err.min().item():.1e} "
                  f"#{err.median().item():.1e} "
                  f"-{err.mean().item():.1e} "
                  f"^{max_:.1e}",
                  k)
