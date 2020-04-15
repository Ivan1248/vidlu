import torch
import numpy as np
import matplotlib.pyplot as plt

pre_cs = torch.load('/home/igrubisic/gta_cs_petra/pre_cs.pt')
post_cs = torch.load('/home/igrubisic/gta_cs_petra/post_cs.pt')

for stat in ['mean', 'var']:
    def filter_statistic(state):
        return {k: v for k, v in state.items() if f'running_{stat}' in k}


    for k in pre_cs:
        print(k)

    print("\n\n\n")

    pre_cs_f = filter_statistic(pre_cs)
    post_cs_F = filter_statistic(post_cs)

    for k in pre_cs:
        print(k)

    distances = np.array(
        [(a - b).abs().mean().item() for a, b in zip(pre_cs_f.values(), post_cs_F.values())])
    plt.plot(np.arange(len(distances)), distances, label=stat)
    # distances = np.array([(a-b).abs().max().item() for a, b in zip(pre_cs_f.values(), post_cs_F.values())])
    # plt.plot(np.arange(len(distances)), distances, label=stat+'max')

# plt.xticks(list(pre_cs.keys()))
plt.legend()
plt.show()

# embed()
