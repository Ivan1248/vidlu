Running these scripts should reproduce experiments from "[A baseline for semi-supervised learning of efficient semantic segmentation models](https://arxiv.org/abs/2106.07075)". Some other data can be found [here](https://github.com/Ivan1248/semisup-seg-efficient),


# Configuration check

This runs 3 configurations for testing whether everything is set up properly:
```sh
bash test_setup.sh
```
The `CUDA_VISIBLE_DEVICES` environment variable can be used to choose a GPU.

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
