# ViDLU: Vision Deep Learning Utilities

A deep learning library for research with emphasis on computer vision, based on [PyTorch](https://pytorch.org/).

### Things (to be) done:
- [X] data
  - [X] easy data processing (transformations, splitting, joining)
  - [X] data caching in RAM and HDD
  - [X] lazy data loading and processing (e.g. if segmentation labels are not requested, they are not loaded)
  - [X] many datasets (data loaders)
- [X] expressive and high customizability of models (deep arguments)
- [X] models (convolutional) with training algorithms
  - [X] classification models:
    - [X] ResNet, WRN, DenseNet
    - [ ] RevNet, iRevNet, VGG
    - [X] other: BagNet
  - [X] semantic segmentation
    - [X] basic segmentation head and loss
  - [ ] Ladder DenseNet, SwiftNet
  - [ ] variational inference
    - [ ] stochastic model parts
  - [ ] VAE
  - [ ] flow-based generative models
- [X] training
  - [X] logging, checkpoints, resuming interupted training
  - [ ] pretrained parameter loading: ResNet, DenseNet
  - [X] supervised training
  - [ ] adversarial training
    - [X] single step attack
    - [ ] PGD/BIM attack
    - [ ] CW attack
  - [ ] GAN training
