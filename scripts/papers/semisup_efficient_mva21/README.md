# Configuration check

To test whether everything is set up properly, you can run runs 3 training configurations for 1 epoch with
```sh
bash test_setup.sh
```
You can use the `CUDA_VISIBLE_DEVICES` environment variable to choose a GPU.

# Experiments

Experiments on half-resolution Cityscapes with different proportions of labels (Table 1):
```sh
bash experiments_label_proportions.sh
```

Consistency variant comparison (Table 2):
```sh
bash experiments_cons_variants.sh
```

Qualitative results (Figure 1 in appendix):
```sh
bash generate_images.sh
```
