import torch


def average_adversarial_example(x, y, model, class_count, attack):
    x_advs = []
    for c in range(class_count):
        x_advs.append(attack.perturb(x, torch.full_like(y, c)))
    return torch.mean(torch.stack(x_advs), 0)
