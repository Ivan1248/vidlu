import torch


def mean_adversarial_example(x, y, class_count, attack):
    x_advs = []
    for c in range(class_count):
        x_advs.append(attack.perturb(x, torch.full_like(y, c)))
    return torch.mean(torch.stack(x_advs), 0)


def mean_adversarial_example_prediction(x, y, model, class_count, attack):
    x_advs, preds = [], []
    for c in range(class_count):
        x_advs.append(attack.perturb(x, torch.full_like(y, c)))
        preds.append(model(x_advs[-1]))
    return torch.mean(torch.stack(preds), 0)


def max_adversarial_example_prediction(x, y, model, class_count, attack):
    x_advs, preds = [], []
    for c in range(class_count):
        x_advs.append(attack.perturb(x, torch.full_like(y, c)))
        preds.append(model(x_advs[-1]))
    return torch.max(torch.stack(preds), 0)
