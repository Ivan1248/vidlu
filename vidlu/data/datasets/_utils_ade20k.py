# From: https://github.com/CSAILVision/ADE20K/blob/d31f5a1a7a8baf3f24d9e0200610d437ef7b0921/utils/utils_ade20k.py
# Modified.

from PIL import Image
import matplotlib._color_data as mcd
import cv2
import json
import numpy as np
import os

_NUMERALS = '0123456789abcdefABCDEF'
_HEXDEC = {v: int(v, 16) for v in (x + y for x in _NUMERALS for y in _NUMERALS)}
LOWERCASE, UPPERCASE = 'x', 'X'


def rgb(triplet):
    return _HEXDEC[triplet[0:2]], _HEXDEC[triplet[2:4]], _HEXDEC[triplet[4:6]]


def loadAde20K(file):
    fileseg = file.replace('.jpg', '_seg.png');
    with Image.open(fileseg) as io:
        seg = np.array(io);

    # Obtain the segmentation mask, bult from the RGB channels of the _seg file
    R = seg[:, :, 0];
    G = seg[:, :, 1];
    B = seg[:, :, 2];
    ObjectClassMasks = (R / 10).astype(np.int32) * 256 + (G.astype(np.int32));

    # Obtain the instance mask from the blue channel of the _seg file
    Minstances_hat = np.unique(B, return_inverse=True)[1]
    Minstances_hat = np.reshape(Minstances_hat, B.shape)
    ObjectInstanceMasks = Minstances_hat

    level = 0
    PartsClassMasks = [];
    PartsInstanceMasks = [];
    while True:
        level = level + 1;
        file_parts = file.replace('.jpg', '_parts_{}.png'.format(level));
        if os.path.isfile(file_parts):
            with Image.open(file_parts) as io:
                partsseg = np.array(io);
            R = partsseg[:, :, 0];
            G = partsseg[:, :, 1];
            B = partsseg[:, :, 2];
            PartsClassMasks.append((np.int32(R) / 10) * 256 + np.int32(G));
            PartsInstanceMasks = PartsClassMasks
            # TODO:  correct partinstancemasks


        else:
            break

    objects = {}
    parts = {}

    attr_file_name = file.replace('.jpg', '.json')
    if os.path.isfile(attr_file_name):
        with open(attr_file_name, 'r') as f:
            input_info = json.load(f)

        contents = input_info['annotation']['object']
        instance = np.array([int(x['id']) for x in contents])
        names = [x['raw_name'] for x in contents]
        corrected_raw_name = [x['name'] for x in contents]
        partlevel = np.array([int(x['parts']['part_level']) for x in contents])
        ispart = np.array([p > 0 for p in partlevel])
        iscrop = np.array([int(x['crop']) for x in contents])
        listattributes = [x['attributes'] for x in contents]
        polygon = [x['polygon'] for x in contents]
        for p in polygon:
            p['x'] = np.array(p['x'])
            p['y'] = np.array(p['y'])

        objects['instancendx'] = instance[ispart == 0]
        objects['class'] = [names[x] for x in list(np.where(ispart == 0)[0])]
        objects['corrected_raw_name'] = [corrected_raw_name[x] for x in
                                         list(np.where(ispart == 0)[0])]
        objects['iscrop'] = iscrop[ispart == 0]
        objects['listattributes'] = [listattributes[x] for x in list(np.where(ispart == 0)[0])]
        objects['polygon'] = [polygon[x] for x in list(np.where(ispart == 0)[0])]

        parts['instancendx'] = instance[ispart == 1]
        parts['class'] = [names[x] for x in list(np.where(ispart == 1)[0])]
        parts['corrected_raw_name'] = [corrected_raw_name[x] for x in
                                       list(np.where(ispart == 1)[0])]
        parts['iscrop'] = iscrop[ispart == 1]
        parts['listattributes'] = [listattributes[x] for x in list(np.where(ispart == 1)[0])]
        parts['polygon'] = [polygon[x] for x in list(np.where(ispart == 1)[0])]

    return {'img_name': file, 'segm_name': fileseg,
            'class_mask': ObjectClassMasks, 'instance_mask': ObjectInstanceMasks,
            'partclass_mask': PartsClassMasks, 'part_instance_mask': PartsInstanceMasks,
            'objects': objects, 'parts': parts}


def plot_polygon(img_name, info, show_obj=True, show_parts=False):
    colors = mcd.CSS4_COLORS
    color_keys = list(colors.keys())
    all_objects = []
    all_poly = []
    if show_obj:
        all_objects += info['objects']['class']
        all_poly += info['objects']['polygon']
    if show_parts:
        all_objects += info['parts']['class']
        all_poly += info['objects']['polygon']

    img = cv2.imread(img_name)
    thickness = 5
    for it, (obj, poly) in enumerate(zip(all_objects, all_poly)):
        curr_color = colors[color_keys[it % len(color_keys)]]
        pts = np.concatenate([poly['x'][:, None], poly['y'][:, None]], 1)[None, :]
        color = rgb(curr_color[1:])
        img = cv2.polylines(img, pts, True, color, thickness)
    return img


NUM_CLASSES = 150

colors = mcd.CSS4_COLORS
colors_keys = list(colors.keys())
CLASS_COLORS = [rgb(mcd.CSS4_COLORS[colors_keys[i % len(colors_keys)]][1:]) for i in range(NUM_CLASSES)]
CLASS_NAMES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed ", "windowpane", "grass",
    "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant", "curtain",
    "chair", "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field",
    "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub", "railing",
    "cushion", "base", "box", "column", "signboard", "chest of drawers", "counter", "sand", "sink",
    "skyscraper", "fireplace", "refrigerator", "grandstand", "path", "stairs", "runway", "case",
    "pool table", "pillow", "screen door", "stairway", "river", "bridge", "bookcase", "blind",
    "coffee table", "toilet", "flower", "book", "hill", "bench", "countertop", "stove", "palm",
    "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus",
    "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth",
    "television receiver", "airplane", "dirt track", "apparel", "pole", "land", "bannister",
    "escalator", "ottoman", "bottle", "buffet", "poster", "stage", "van", "ship", "fountain",
    "conveyer belt", "canopy", "washer", "plaything", "swimming pool", "stool", "barrel", "basket",
    "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball", "food", "step", "tank",
    "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher", "screen",
    "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray", "ashcan", "fan",
    "pier", "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator", "glass",
    "clock", "flag"]
# Colors from https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/datasets/ade.py
CLASS_COLORS = [(120, 120, 120), (180, 120, 120), (6, 230, 230), (80, 50, 50), (4, 200, 3),
                (120, 120, 80), (140, 140, 140), (204, 5, 255), (230, 230, 230), (4, 250, 7),
                (224, 5, 255), (235, 255, 7), (150, 5, 61), (120, 120, 70), (8, 255, 51),
                (255, 6, 82), (143, 255, 140), (204, 255, 4), (255, 51, 7), (204, 70, 3),
                (0, 102, 200), (61, 230, 250), (255, 6, 51), (11, 102, 255), (255, 7, 71),
                (255, 9, 224), (9, 7, 230), (220, 220, 220), (255, 9, 92), (112, 9, 255),
                (8, 255, 214), (7, 255, 224), (255, 184, 6), (10, 255, 71), (255, 41, 10),
                (7, 255, 255), (224, 255, 8), (102, 8, 255), (255, 61, 6), (255, 194, 7),
                (255, 122, 8), (0, 255, 20), (255, 8, 41), (255, 5, 153), (6, 51, 255),
                (235, 12, 255), (160, 150, 20), (0, 163, 255), (140, 140, 140), (250, 10, 15),
                (20, 255, 0), (31, 255, 0), (255, 31, 0), (255, 224, 0), (153, 255, 0), (0, 0, 255),
                (255, 71, 0), (0, 235, 255), (0, 173, 255), (31, 0, 255), (11, 200, 200),
                (255, 82, 0), (0, 255, 245), (0, 61, 255), (0, 255, 112), (0, 255, 133),
                (255, 0, 0), (255, 163, 0), (255, 102, 0), (194, 255, 0), (0, 143, 255),
                (51, 255, 0), (0, 82, 255), (0, 255, 41), (0, 255, 173), (10, 0, 255),
                (173, 255, 0), (0, 255, 153), (255, 92, 0), (255, 0, 255), (255, 0, 245),
                (255, 0, 102), (255, 173, 0), (255, 0, 20), (255, 184, 184), (0, 31, 255),
                (0, 255, 61), (0, 71, 255), (255, 0, 204), (0, 255, 194), (0, 255, 82),
                (0, 10, 255), (0, 112, 255), (51, 0, 255), (0, 194, 255), (0, 122, 255),
                (0, 255, 163), (255, 153, 0), (0, 255, 10), (255, 112, 0), (143, 255, 0),
                (82, 0, 255), (163, 255, 0), (255, 235, 0), (8, 184, 170), (133, 0, 255),
                (0, 255, 92), (184, 0, 255), (255, 0, 31), (0, 184, 255), (0, 214, 255),
                (255, 0, 112), (92, 255, 0), (0, 224, 255), (112, 224, 255), (70, 184, 160),
                (163, 0, 255), (153, 0, 255), (71, 255, 0), (255, 0, 163), (255, 204, 0),
                (255, 0, 143), (0, 255, 235), (133, 255, 0), (255, 0, 235), (245, 0, 255),
                (255, 0, 122), (255, 245, 0), (10, 190, 212), (214, 255, 0), (0, 204, 255),
                (20, 0, 255), (255, 255, 0), (0, 153, 255), (0, 41, 255), (0, 255, 204),
                (41, 0, 255), (41, 255, 0), (173, 0, 255), (0, 245, 255), (71, 0, 255),
                (122, 0, 255), (0, 255, 184), (0, 92, 255), (184, 255, 0), (0, 133, 255),
                (255, 214, 0), (25, 194, 194), (102, 255, 0), (92, 0, 255)]
