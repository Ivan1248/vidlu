# ViDLU: Vision Deep Learning Utilities

A deep learning library for research with emphasis on computer vision, based on [PyTorch](https://pytorch.org/).

[![Build Status](https://travis-ci.org/Ivan1248/Vidlu.svg?branch=master)](https://travis-ci.org/Ivan1248/Vidlu)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/7f89c65e677f490bab26c0e5c7cae116)](https://www.codacy.com/manual/Ivan1248/Vidlu?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Ivan1248/Vidlu&amp;utm_campaign=Badge_Grade)

## Things (to be) done
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
  - [X] models (convolutional) with training algorithms
    - [X] classification
      - [x] ResNet, WRN, DenseNet
      - [ ] RevNet, iRevNet, VGG
      - [x] other: BagNet
    - [X] semantic segmentation
      - [x] basic segmentation head and loss
      - [x] SwiftNet
      - [ ] Ladder DenseNet
    - [ ] variational inference
      - [ ] stochastic model parts
    - [ ] VAE
    - [ ] flow generative models
  - [x] training
    - [x] inductive bias (mostly) disentangled from the model
    - [x] customizable training (inductive bias) configurations
    - [x] logging, checkpoints, resuming interrupted training
    - [X] pretrained parameter loading: ResNet, DenseNet
    - [x] supervised training
    - [x] adversarial training
    - [X] semi-supervised training (VAT)
    - [ ] GAN training
    - [ ] natural gradient variational inference
  - adversarial attacks
    - [x] single step gradient
    - [x] PGD/BIM
      - [x] stop on success
      - [x] support for "adversarial training for free"
    - [X] VAT attack
    - [ ] CW
    - [ ] DDN (arXiv:1811.09600)
    - [ ] Frank-Wolfe constrained optimization

<!--
In many places in the code some parameter names end with "_f". 
This means that the argument is not a final object but a factory (hence "_f") 
that produces an object, e.g. "backbone_f()" should produce a "backbone". This 
is to allow more flexibility e.g.

```
def foo(..., bar_f=make_some_bar):
    ...

def make_some_bar(..., baz_f=make_some_baz):
    ...
    
def make_some_baz(..., swallow_type='european'):
    ...

t=ArgTree
argtree_partial(foo, bar_f=ArgTree(baz_f=t(swallow_type='african)))
```

instead of

```
def foo(..., bar_args, baz_args):
    make_some_bar(..., **bar_args, **baz_args)
    ...

def make_some_bar(..., **baz_args):
    make_some_baz(..., **baz_args)
    ...
    
def make_some_baz(..., swallow_type='european'):
    ...

foo(baz_args=dict(swallow_type='african'))
```
-->
