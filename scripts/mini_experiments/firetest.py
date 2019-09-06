import fire

def run(data: str, model: str, trainer: str, evaluation: str = None, device='', experiment_suffix='', resume=False,
        restart=False, seed=53, verbosity=1):
    """
    Args:
        data: adsf
        model: sdsf
        trainer: dsffsdfdsdfdsfd
        evaluation: df
        device: sdfdsfsdsfdf
        experiment_suffix: fdsfsfd
        resume: fdsfdsfd
        restart: dsffdsfsd
        seed: fdfsdf d
        verbosity: f df fd s
    """
    print(locals())

@torch.no_grad()
def rand_init_delta(delta, ord, eps):
    if isinstance(eps, torch.Tensor):
        assert len(eps) == len(delta)

    delta.uniform_(-1, 1) * delta.numel() ** (1 / ord)
    if ord == np.inf:
        delta.mul_(eps)
    elif ord in [1, 2]:
        delta = batchops.restrict_norm(delta, eps, ord)
    else:
        error = "Only ord = inf and ord = 2 have been implemented"
        raise NotImplementedError(error)

    return delta

def main():
    delta=torch.zeros(100)

fire.Fire(main)