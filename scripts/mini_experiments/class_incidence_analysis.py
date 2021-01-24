import pickle
import numpy as np
from tqdm import tqdm
import IPython

dataset = 'ADE20k'
incidences = np.load(f'/home/morsic/datasets/{dataset}/class_incidence_train.npy')
with open(f'/home/morsic/datasets/{dataset}/class_instances_train.pkl', 'rb') as f:
    instances = pickle.load(f)

incidences = incidences[:-1]  # remove ignore id
num_images = len(instances)
num_classes = incidences.shape[0]
present_in_image = np.zeros((num_images, num_classes), dtype=np.uint32)
image_names = np.array(list(instances.keys()))

for i, (k, v) in enumerate(tqdm(instances.items(), total=len(instances))):
    for idx in v.keys():
        if idx >= num_classes:
            continue
        present_in_image[i, idx] += len(v[idx])

class_incidence_histogram = incidences / incidences.sum()
indices_by_occurence = np.argsort(class_incidence_histogram)
is_image_chosen = np.zeros(num_images, dtype=np.bool)
chosen_class = num_classes * np.ones(num_images, dtype=np.uint32)

p_r = class_incidence_histogram.sum() / class_incidence_histogram
p_r /= p_r.sum()
images_to_sample = np.round(num_images * p_r).astype(np.uint32)

for idx in indices_by_occurence:
    possibilities = np.where(present_in_image[:, idx] > 0 & ~is_image_chosen)[0]
    to_sample = min(images_to_sample[idx], len(possibilities))
    # print(f'{to_sample:4d} / {images_to_sample[idx]:4d}')
    chosen = np.random.choice(possibilities, to_sample)
    is_image_chosen[chosen] = 1
    chosen_class[chosen] = idx

chosen_classes = {}
for n, c in zip(image_names, chosen_class):
    chosen_classes[n] = c
# print(f'{is_image_chosen.sum()} sampled uniformly our of {num_images}')
IPython.embed()
