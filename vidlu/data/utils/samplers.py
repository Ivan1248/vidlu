import torch.utils.data as tud


def multiset_sampler(multiplicities, dataset=None):
    if dataset is not None:
        if multiplicities is None:
            multiplicities = dataset.info['multiplicities']
        if not len(dataset) == len(multiplicities):
            raise RuntimeError(f"The size of multiplicities ({len(multiplicities)} does not equal"
                               + f" the size of the dataset ({len(dataset)}).")
    indices = [i for i, m in enumerate(multiplicities) for _ in range(m)]
    return tud.SubsetRandomSampler(indices=indices)
