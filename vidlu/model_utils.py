import datetime

import tensorflow as tf

from vidlu.learning.models import Model, ModelDef
from vidlu.learning.models import BlockStructure, InferenceComponent
from vidlu.learning.models import Layers, TrainingComponents
from .evaluation import ClassificationEvaluator
from . import dirs, parameter_loading


class StandardFeatureExtractors:

    @staticmethod
    def resnet(depth, cifar_root_block, base_width, dropout, non_trainable_groups=[]):
        assert not dropout
        print(f'ResNet-{depth}-{base_width}')
        normal = ([3, 3], [1, 1], 'id')
        bottleneck = ([1, 3, 1], [1, 1, 4], 'proj')  # last paragraph in [2]
        group_lengths, (ksizes, width_factors, dim_change) = {
            18: ([2] * 4, normal),  # [1] bw 64
            34: ([3, 4, 6, 3], normal),  # [1] bw 64
            110: ([18] * 3, normal),  # [1] bw 16
            50: ([3, 4, 6, 3], bottleneck),  # [1] bw 64
            101: ([3, 4, 23, 3], bottleneck),  # [1] bw 64
            152: ([3, 8, 36, 3], bottleneck),  # [1] bw 64
            164: ([18] * 3, bottleneck),  # [1] bw 16
            200: ([3, 24, 36, 3], bottleneck),  # [2] bw 64
        }[depth]
        return Layers.Features.resnet(
            base_width=base_width,
            group_lengths=group_lengths,
            block_structure=BlockStructure.resnet(
                ksizes=ksizes,
                dropout_locations=[0] if dropout else [],
                width_factors=width_factors),
            dim_change=dim_change,
            cifar_root_block=cifar_root_block,
            non_trainable_groups=non_trainable_groups)

    @staticmethod
    def wide_resnet(depth,
                    width_factor,
                    cifar_root_block,
                    dropout,
                    dim_change='proj', non_trainable_groups=set()):
        print(f'WRN-{depth}-{width_factor}')
        zagoruyko_depth = depth
        group_count, ksizes = 3, [3, 3]
        group_depth = (group_count * len(ksizes))
        blocks_per_group = (zagoruyko_depth - 4) // group_depth
        depth = blocks_per_group * group_depth + 4
        assert zagoruyko_depth == depth, \
            f"Invalid depth = {zagoruyko_depth} != {depth} = zagoruyko_depth"
        d = {}
        if type(dropout) in [int, float]:
            d['dropout_rate'] = dropout
        elif dropout:
            d['dropout_rate'] = 0.3
        return Layers.Features.resnet(
            base_width=16,
            group_lengths=[blocks_per_group] * group_count,
            block_structure=BlockStructure.resnet(
                ksizes=ksizes, dropout_locations=[0] if dropout else []),
            width_factor=width_factor,
            dim_change=dim_change,
            cifar_root_block=cifar_root_block,
            **d,
            non_trainable_groups=non_trainable_groups)

    @staticmethod
    def densenet(depth, base_width, cifar_root_block, dropout):
        # dropout if no data augmentation
        print(f'DenseNet-{depth}-{base_width}')
        ksizes = [1, 3]
        depth_to_group_lengths = {
            121: [6, 12, 24, 16],  # base_width = 32
            161: [6, 12, 36, 24],  # base_width = 48
            169: [6, 12, 32, 32],  # base_width = 32
        }
        if depth in depth_to_group_lengths:
            group_lengths = depth_to_group_lengths[depth]
        else:
            group_count = 3
            assert (depth - group_count - 1) % 3 == 0, \
                f"invalid depth: (depth-group_count-1) must be divisible by 3"
            blocks_per_group = (depth - group_count - 1) // \
                               (group_count * len(ksizes))
            group_lengths = [blocks_per_group] * group_count
        return Layers.Features.densenet(
            base_width=base_width,
            group_lengths=group_lengths,
            block_structure=BlockStructure.densenet(ksizes=ksizes),
            cifar_root_block=cifar_root_block,
            dropout_rate=0.2 if dropout else 0)

    @staticmethod
    def ladder_densenet(depth, base_width, dropout, cifar_root_block=False):
        print(f'Ladder-DenseNet-{depth}')
        group_lengths = {
            121: [6, 12, 24, 16],  # base_width = 32
            161: [6, 12, 36, 24],  # base_width = 48
            169: [6, 12, 32, 32],  # base_width = 32
        }[depth]
        return Layers.Features.ladder_densenet(
            base_width=32,
            cifar_root_block=cifar_root_block,
            group_lengths=group_lengths,
            dropout_rate=0.2 if dropout else 0)


def get_inference_component(
        net_name,
        ds_train,
        depth: int,
        dropout: bool,
        aleatoric_uncertainty=False,
        base_width: int = None,  # rn, dn, ldn
        width_factor: int = None,
        special_features: set = None):
    assert net_name in ['rn', 'wrn', 'dn', 'ldn']
    assert bool(base_width) == (net_name in ['rn', 'dn', 'ldn'])
    assert bool(width_factor) == (net_name == 'wrn')

    problem_id = ds_train.info['problem_id']

    layers = []
    add = layers.append

    # input to features
    sfe = StandardFeatureExtractors
    sfe_args = {
        'depth':
            depth,
        'cifar_root_block':
            ds_train.info['id'] in ['cifar', 'cifar100', 'svhn', 'tinyimagenet'],
        'dropout':
            dropout,
        'non_trainable_groups': [] if 'binary_outlier_detection' in special_features else []
    # 0,1,2
    }
    if net_name in ['rn', 'dn', 'ldn']:
        sfe_args['base_width'] = base_width
    input_to_features_layer = {
        'rn': sfe.resnet,
        'wrn': lambda **k: sfe.wide_resnet(**k, width_factor=width_factor),
        'dn': sfe.densenet,
        'ldn': sfe.ladder_densenet,
    }[net_name]
    add(input_to_features_layer(**sfe_args))

    # logits
    class_count = ds_train.info['class_count']
    if (problem_id, net_name) == ('semseg', 'ldn'):
        add(Layers.Logits.segmentation(class_count))
        add(lambda x, **k: (x[0], {}))  # extract main logits
    elif problem_id == 'semseg':
        if aleatoric_uncertainty:
            add(Layers.Logits.segmentation_gaussian(class_count))
        else:
            add(Layers.Logits.segmentation(class_count))
    elif problem_id in 'clf':
        assert not aleatoric_uncertainty, "Not implemented"
        if 'devries_confidence' in special_features:
            add(Layers.Logits.devries_confidence_classification(class_count))
        elif 'petra_confidence' in special_features:
            add(Layers.Logits.petra_confidence_classification(class_count))
        else:
            add(Layers.Logits.classification(class_count))
    else:
        assert False, f"Not avaliable: problem_id={problem_id}, net_name={net_name}"

    # output
    if problem_id in ['semseg', 'clf']:
        if aleatoric_uncertainty:
            add(Layers.Output.argmax_and_probs_from_gaussian_logits())
        else:
            if 'devries_confidence' in special_features or 'petra_confidence' in special_features:
                add(Layers.Output.devries_argmax_and_probs_and_confidence())
            else:
                add(Layers.Output.argmax_and_probs_from_logits())
        add(Layers.ProbsUncertainty.entropy())
        add(Layers.pull_node('output'))

    return InferenceComponent(Layers.sequence(layers))


def get_training_component(net_name, ds_train, epoch_count, pretrained=False,
                           pretrained_lr_factor=None, special_features=None):
    if pretrained_lr_factor is None:
        pretrained_lr_factor = 1 / 5
    if not pretrained:
        pretrained_lr_factor = 1
    special_features = special_features or []
    problem_id, ds_id = ds_train.info['problem_id'], ds_train.info['id']
    if problem_id == 'clf':
        base_learning_rate = 1e-1
        resnet_learning_rate_policy = {
            'boundaries': [
                int(i * epoch_count / 200 + 0.5) for i in [60, 120, 160]
            ],
            'values': [base_learning_rate * 0.2 ** i for i in range(4)]
        }
        densenet_learning_rate_policy = {
            'boundaries': [int(i * epoch_count / 100 + 0.5) for i in [50, 75]],
            'values': [base_learning_rate * 0.1 ** i for i in range(3)]
        }
        if ds_id in ['cifar', 'cifar100', 'svhn']:
            batch_size = 128
        elif ds_id == 'mozgalo':
            batch_size = 64
        elif ds_id == 'inaturalist18':
            batch_size = 16
        else:
            assert False
        if net_name == 'dn':
            batch_size //= 2
        weight_decay = {'dn': 1e-4,
                        'rn': 1e-4,
                        'wrn': 5e-4}[net_name]
        if "outlier_exposure" in special_features:
            assert ds_id in ['cifar', 'cifar100']
            return TrainingComponents.outlier_exposure(
                batch_size=batch_size,
                weight_decay=weight_decay, epoch_count=epoch_count)

        learning_rate_policy = densenet_learning_rate_policy if net_name == 'dn' else resnet_learning_rate_policy

        return TrainingComponents.standard(
            batch_size=batch_size,
            loss=problem_id,
            weight_decay=weight_decay,
            optimizer=lambda lr: tf.train.MomentumOptimizer(lr, 0.9),
            learning_rate_policy=learning_rate_policy,
            pretrained_lr_factor=pretrained_lr_factor)
    elif problem_id == 'semseg':  # TODO: loss for non ladder-densenet
        batch_size = {
            'cityscapes': 4,
            'voc2012': 4,
            'camvid': 8,
            'iccv09': 16,
        }[ds_id]
        weight_decay = {
            'ldn': 1e-4,  # ladder-densenet/voc2012/densenet.py
            'dn': 1e-4,  # ladder-densenet/voc2012/densenet.py
            'rn': 1e-4,  # ladder-densenet/voc2012/resnet.py
            'wrn': 5e-4
        }[net_name]
        return TrainingComponents.ladder_densenet(
            epoch_count=epoch_count,
            base_learning_rate=5e-4,
            batch_size=batch_size,
            weight_decay=weight_decay,
            pretrained_lr_factor=pretrained_lr_factor,
            loss_weights=[0.7, 0.3] if net_name == 'ldn' else [1])
    else:
        assert False, "Not implemented"


def get_model(net_name,
              ds_train,
              depth,
              width,
              dropout,
              pretrained=False,
              pretrained_lr_factor=None,
              epoch_count=None,
              special_features=[]):
    # width: width factor for WRN, base_width for others
    ic = get_inference_component(
        net_name=net_name,
        ds_train=ds_train,
        depth=depth,  # all
        base_width=None or net_name != 'wrn' and width,
        width_factor=None or net_name == 'wrn' and width,
        dropout=dropout, special_features=special_features)

    tc = get_training_component(
        net_name=net_name,
        ds_train=ds_train,
        epoch_count=epoch_count,
        pretrained=pretrained,
        pretrained_lr_factor=pretrained_lr_factor,
        special_features=special_features)

    problem_id = ds_train.info['problem_id']
    if problem_id in ['clf', 'semseg']:
        ae = ClassificationEvaluator(ds_train.info['class_count'])
    else:
        assert False, "Not implemented"

    model = Model(
        modeldef=ModelDef(ic, tc),
        training_log_period=len(ds_train) // tc.batch_size // 5,
        accumulating_evaluator=ae)

    if pretrained:
        print("Loading pretrained parameters...")
        if net_name == 'rn' and depth == 50 and width == 64:
            names_to_params = parameter_loading.get_resnet_parameters_from_checkpoint_file(
                f'{dirs.PRETRAINED}/resnetv2_50/resnet_v2_50.ckpt',
                include_first_conv=ds_train.info['id'] not in ['cifar'])
        elif net_name == 'dn' and depth == 121 and width == 32:
            names_to_params = parameter_loading.get_densenet_parameters_from_checkpoint_file(
                f'{dirs.PRETRAINED}/densenet_121/tf-densenet121.ckpt')
        elif net_name == 'ldn' and depth == 121 and width == 32:
            names_to_params = parameter_loading.get_ladder_densenet_parameters_from_checkpoint_file(
                f'{dirs.PRETRAINED}/densenet_121/tf-densenet121.ckpt')
        else:
            assert False, "Pretrained parameters not available."
        model.load_parameters(names_to_params)

    return model


def save_trained_model(model,
                       ds_id,
                       net_name,
                       epoch_count,
                       dropout=None,
                       pretrained=None,
                       saved_nets_dir=dirs.SAVED_NETS):
    if dropout:
        net_name += '-do'
    if pretrained:
        net_name += '-pretrained'
    model.save_state(f'{saved_nets_dir}/{ds_id}/' +
                     f'{net_name}-e{epoch_count}/' +
                     f'{datetime.datetime.now():%Y-%m-%d-%H%M}')


def load_trained_model(model,
                       ds_id,
                       net_name,
                       epoch_count,
                       date_code,
                       dropout=None,
                       pretrained=None,
                       saved_nets_dir=dirs.SAVED_NETS):
    if dropout:
        net_name += '-do'
    if pretrained:
        net_name += '-pretrained'
    model.load_state(f'{saved_nets_dir}/{ds_id}/' +
                     f'{net_name}-e{epoch_count}/' + f'{date_code}/Model')
