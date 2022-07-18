from collections import defaultdict

import torchvision
import torch

# noinspection PyUnresolvedReferences
import _context
import dirs

from vidlu import factories, data, modules
from vidlu.factories import problem

# data = data.datasets.HBlobs(example_shape=(32, 32, 3))

dataset = factories.get_data("TinyImageNet{val}", dirs.datasets)[0][0]
dataset = factories.prepare_dataset(dataset)
inp = next(iter(data.DataLoader(dataset, batch_size=1)))[0]

resnet_my = factories.get_model(
    model_str='ResNetV1,backbone_f=t(depth=50,block_f=t(act_f=t(inplace=True)))',
    input_adapter_str='id', problem=problem.Classification(dataset.info.class_count),
    init_input=inp)
resnet_my.eval()

resnet_tv = torchvision.models.resnet50(num_classes=dataset.info.class_count)
state, _ = factories.get_translated_parameters("resnet", state_dict=resnet_tv.state_dict())
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
    'layer1.0.conv1':
        'backbone.bulk.unit0_0.fork.block.conv0',
    'layer1.0.bn1':
        'backbone.bulk.unit0_0.fork.block.norm0',
    'layer1.0':
        'backbone.bulk.unit0_0',
    'layer1.1':
        'backbone.bulk.unit0_1',
    'layer2.0.conv1':
        'backbone.bulk.unit1_0.fork.block.conv0',  # stride 2 after 1x1 conv.
    'layer2.0':
        'backbone.bulk.unit1_0',
    'layer4':
        'backbone.bulk',
    'avgpool':
        'head.pre_logits_mean',
    'fc':
        'head.logits',
}

output_pairs = defaultdict(list)

for tv, my in layer_pairs.items():
    def record_input(module, inp, output, k=tv):
        output_pairs[k].append(output)


    modules.get_submodule(resnet_tv, tv).register_forward_hook(record_input)
    modules.get_submodule(resnet_my, my).register_forward_hook(record_input)

my_state = resnet_my.state_dict()

for k, v in resnet_my.named_modules():
    print(k)
print()
for k, v in resnet_tv.named_modules():
    print(k)
print()

print("Parameter/buffer equivalence:")
with torch.no_grad():
    for k in state.keys():
        my, tv = my_state[k], state[k]
        if len(my.shape) == len(tv.shape) == 0:
            print(f'[]={k}')

        # err = (my - tv).abs() / (torch.min(my.abs() + tv.abs()) + 1e-16)
        err = torch.any(my != tv).float()
        max_ = err.max().item()
        if max_ == 0:
            print('OK', k)
        else:
            print(f"_{err.min().item():.1e} "
                  f"#{err.median().item():.1e} "
                  f"-{err.mean().item():.1e} "
                  f"^{max_:.1e}",
                  k)
print()

print("Activation equivalence:")
output_my = resnet_my(inp)
output_tv = resnet_tv(inp)
for k, (my, tv) in output_pairs.items():
    err = (my - tv).abs() / (torch.min(my.abs() + tv.abs()) + 1e-16)

    print(f"_{err.min().item():.1e} #{err.median().item():.1e}"
          + f" -{err.mean().item():.1e} ^{err.max().item():.1e}", k)
    # err.min().item(),, err.mean().item()
print()

print("Parameter gradient equivalance:")
output_my.pow(2).sum().backward()
output_tv.pow(2).sum().backward()
grads_my = {k: v.grad for k, v in resnet_my.named_parameters()}
grads_tv = {k: v.grad for k, v in resnet_tv.named_parameters()}
grads_tv, _ = factories.get_translated_parameters("resnet", state_dict=grads_tv)
with torch.no_grad():
    for k in grads_my.keys():
        my, tv = grads_my[k], grads_tv[k]
        if my is None or tv is None:
            if my is not tv:
                print(f"Only one is none: {my}, {tv}")
            continue
        if len(my.shape) == len(tv.shape) == 0:
            print(f'[]={k}')

        # err = (my - tv).abs() / (torch.min(my.abs() + tv.abs()) + 1e-16)
        err = torch.any(my != tv).float()
        max_ = err.max().item()
        if max_ == 0:
            print('OK', k)
        else:
            print(f"_{err.min().item():.1e} "
                  f"#{err.median().item():.1e} "
                  f"-{err.mean().item():.1e} "
                  f"^{max_:.1e}",
                  k)
