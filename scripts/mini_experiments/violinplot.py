import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import seaborn as sns

sns.set()

data = list(map(np.array, pickle.load(open("layer_del_spp.pkl", 'rb')).values()))

seaborn.violinplot(data=[[0]] + data, width=1, cut=0, color="orange")

# plt.violinplot(data, showmeans=True, showextrema=True, showmedians=True)
