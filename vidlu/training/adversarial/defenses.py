import torch


def mean_adversarial_example(x, y, class_count, attack):
    x_advs = [attack.perturb(x, torch.full_like(y, c)) for c in range(class_count)]
    return torch.mean(torch.stack(x_advs), 0)


def get_all_direction_predictions(x, y, model, class_count, attack):
    return [model(attack.perturb(x, torch.full_like(y, c))) for c in range(class_count)]


def mean_adversarial_example_prediction(x, y, model, class_count, attack):
    preds = get_all_direction_predictions(x, y, model, class_count, attack)
    return torch.mean(torch.stack(preds), 0)


def max_adversarial_example_prediction(x, y, model, class_count, attack):
    preds = get_all_direction_predictions(x, y, model, class_count, attack)
    return torch.max(torch.stack(preds), 0)
