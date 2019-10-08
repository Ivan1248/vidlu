# ViDLU: Vision Deep Learning Utilities

A deep learning library for research with emphasis on computer vision, based on [PyTorch](https://pytorch.org/).

## Things (to be) done

- general
  - [x] data
    - [x] easy data processing (transformations, splitting, joining)
    - [x] data caching in RAM and HDD
    - [x] lazy data loading and processing (e.g. if segmentation labels are not requested, they are not loaded)
    - [x] many datasets (data loaders)
  - [x] modules 
    - [X] modules that infer their input dimension automatically (like in e.g. [MXNet](http://mxnet.incubator.apache.org/) and
        [MagNet](https://github.com/MagNet-DL/magnet)) -- requires an initial run for initialization
    - [X] procedures for getting module names (paths) from the call stack and extracting intermediate layer outputs through their names
    - [X] high generality/customizability of components and models (deep arguments) (not completely satisfactory) 
    - [X] deep module splitting and joining (implemented for `vidlu.modules.Sequential`) 
    - [X] extraction of intermediate outputs without changing the module (`vidlu.modules.with_intermediate_outputs`, using `register_forward_hook`)
  - [x] training
    - [x] inductive bias (mostly) disentangled from the model
    - [x] customizable training (inductive bias) configurations
    - [x] logging, checkpoints, resuming interrupted training
    - [X] pretrained parameter loading: ResNet, DenseNet
    - [x] supervised training
    - [x] adversarial training
    - [ ] GAN training
- algorithms
  - models (convolutional) with training algorithms
    - classification
      - [x] ResNet, WRN, DenseNet
      - [ ] RevNet, iRevNet, VGG
      - [x] other: BagNet
    - semantic segmentation
      - [x] basic segmentation head and loss
      - [x] SwiftNet
      - [ ] Ladder DenseNet
    - [ ] variational inference
      - [ ] stochastic model parts
    - [ ] VAE
    - [ ] flow-based generative models
    - [ ] natural gradient variational inference
  - adversarial attacks
    - [x] single step gradient
    - [x] PGD/BIM
      - [x] stop on success
      - [x] free adversarial training
    - [ ] CW
    - [ ] DDN (arXiv:1811.09600)
    - [ ] Frank-Wolfe constrained optimization
