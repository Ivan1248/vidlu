# noinspection PyUnresolvedReferences
import _context

import torch
from pathlib import Path

from tqdm import tqdm, trange

from vidlu import gpu_utils, metrics
from vidlu.training.robustness import attacks
import vidlu.factories as vf
import dirs

device = torch.device(gpu_utils.get_first_available_device(max_gpu_util=0.5, no_processes=False))

data_str = 'cifar10{trainval,test}'
model_str = 'ResNetV2,backbone_f=t(depth=18,small_input=True)'
trainer_str = ('tc.resnet_cifar,tc.adversarial,'
               + 'attack_f=partial(tc.madry_cifar10_attack,step_count=7,stop_on_success=True),'
               + 'eval_attack_f=t(step_count=10,stop_on_success=True)')
data = vf.get_prepared_data_for_trainer(data_str, datasets_dir=dirs.datasets, cache_dir=dirs.cache)
model = vf.get_model('ResNetV2,backbone_f=t(depth=18,small_input=True)', prep_dataset=data.train,
                     device=device)
trainer = vf.get_trainer(trainer_str, dataset=data.train, model=model)
state = torch.load(Path('/home/igrubisic/data/states') / data_str / model_str / trainer_str
                   / '_/200' / 'model_state.pth')
model.load_state_dict(state)

class_count = data.test.info.class_count
probs = [[]] * class_count

attack = attacks.PGDAttack(step_count=4, minimize=True)
metric = metrics.AverageMultiMetric(filter=lambda _: True)
batch_size = 96
for batch in tqdm(trainer.data_loader_f(data.test, batch_size=batch_size, drop_last=True)):
    x, y = trainer.prepare_batch(batch)
    results = [dict() for _ in range(batch_size)]
    probs = []
    for t in trange(class_count):
        t_var = torch.full_like(y, t)
        x_adv = attack.perturb(model, x, t_var)
        prob = trainer.extend_output(model(x_adv))[1].probs[:, t]
        probs.append(prob)
    for i in range(batch_size):
        result = {f'p{t}': probs[t][i].item() for t in range(class_count)}
    metric.update(result)

metric_results = metric.compute()
for k, v in metric_results.items():
    print(f"{k} = {v}")
