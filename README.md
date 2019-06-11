# ViDLU: Vision Deep Learning Utilities

A deep learning library for research with emphasis on computer vision, based on [PyTorch](https://pytorch.org/).

## Things (to be) done

- general
  - [X] data
    - [X] easy data processing (transformations, splitting, joining)
    - [X] data caching in RAM and HDD
    - [X] lazy data loading and processing (e.g. if segmentation labels are not requested, they are not loaded)
    - [X] many datasets (data loaders)
  - [X] modules
    - [X] modules without having to define the input dimension (like in e.g. MXNet or MagNet) -- requires an initial run for initialization
    - [X] procedures for getting module names (paths) from the call stack and extracting intermediate layer outputs through their names
    - [X] high generality/customizability of components and models (deep arguments) (not completely satisfactory)
  - [X] training
    - [X] inductive bias (mostly) disentangled from the model
    - [X] customizable training (inductive bias) configurations
    - [X] logging, checkpoints, resuming interupted training
    - [ ] pretrained parameter loading: ResNet, DenseNet
    - [X] supervised training
    - [X] adversarial training
    - [ ] GAN training
- algorithms
  - models (convolutional) with training algorithms
    - classification
      - [X] ResNet, WRN, DenseNet
      - [ ] RevNet, iRevNet, VGG
      - [X] other: BagNet
    - semantic segmentation
      - [X] basic segmentation head and loss
    - [ ] Ladder DenseNet, SwiftNet
    - [ ] variational inference
      - [ ] stochastic model parts
    - [ ] VAE
    - [ ] flow-based generative models
  - adversarial attacks
    - [X] single step gradient
    - [X] PGD/BIM
    - [ ] CW
    - [ ] DDN (arXiv:1811.09600)
    - [ ] Frank-Wolfe constrained optimization
