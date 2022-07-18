from vidlu.data.class_mapping import invert_many_to_one_mapping
from .datasets import _cityscapes_class_names

# def remaining_taxonomy_part(taxonomy, classes):
#     used_classes = {c for k, subs in taxonomy.items() for c in subs}
#     return {c: [c] for c in classes if c not in used_classes}
#
#
# def complete_taxonomy_mapping(abstractions, classes):
#     return {**abstractions, **remaining_taxonomy_part(abstractions, classes)}


"""
Use invert_many_to_one_mapping from vidlu.data.class_mapping to get one to many mappings.
"""


class Cityscapes:
    identity = {k: k for k in _cityscapes_class_names}
    four_wheel_mapping = {
        **{k: k for k in _cityscapes_class_names},
        **{k: 'four-wheel-vehicle' for k in ['car', 'truck', 'bus']}}
    personal_vehicle_mapping = {
        **{k: k for k in _cityscapes_class_names},
        **{k: 'personal-vehicle' for k in ['car', 'bicycle', 'motorcycle']}}


class Vistas:
    cityscapes_mapping = {
        'construction--barrier--curb': 'sidewalk',
        'construction--barrier--fence': 'fence',
        'construction--barrier--guard-rail': 'fence',
        'construction--barrier--wall': 'wall',
        'construction--flat--bike-lane': 'road',
        'construction--flat--crosswalk-plain': 'road',
        'construction--flat--curb-cut': 'sidewalk',
        'construction--flat--parking': 'road',
        'construction--flat--pedestrian-area': 'sidewalk',
        'construction--flat--rail-track': 'road',
        'construction--flat--road': 'road',
        'construction--flat--service-lane': 'road',
        'construction--flat--sidewalk': 'sidewalk',
        'construction--structure--bridge': 'building',
        'construction--structure--building': 'building',
        'construction--structure--tunnel': 'building',
        'human--person': 'person',
        'human--rider--bicyclist': 'rider',
        'human--rider--motorcyclist': 'rider',
        'human--rider--other-rider': 'rider',
        'marking--crosswalk-zebra': 'road',
        'marking--general': 'road',
        'nature--sand': 'terrain',
        'nature--sky': 'sky',
        'nature--snow': 'terrain',
        'nature--terrain': 'terrain',
        'nature--vegetation': 'vegetation',
        'object--support--pole': 'pole',
        'object--support--traffic-sign-frame': 'traffic sign',
        'object--support--utility-pole': 'pole',
        'object--traffic-light': 'traffic light',
        'object--traffic-sign--front': 'traffic sign',
        'object--vehicle--bicycle': 'bicycle',
        'object--vehicle--bus': 'bus',
        'object--vehicle--car': 'car',
        'object--vehicle--motorcycle': 'motorcycle',
        'object--vehicle--on-rails': 'train',
        'object--vehicle--truck': 'truck',
    }
