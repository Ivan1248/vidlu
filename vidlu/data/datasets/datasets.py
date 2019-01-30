import pickle
import json
from io import BytesIO

import PIL.Image as pimg
import numpy as np
from skimage.transform import resize
from scipy.io import loadmat
from skimage.filters import gaussian as gblur
import torchvision.datasets as dset

from .. import Dataset, Record

from vidlu.utils.image.shape import pad_to_shape, crop
from pathlib import Path

from ._cityscapes_labels import labels as cslabels


# Helper functions


def _load_image(path, force_rgb=True):
    img = pimg.open(path)
    if force_rgb:
        img = img.convert('RGB')
    return img


def _make_example(*args, **kwargs):
    return Record(*args, **kwargs)  # or dict(r.items()) or tuple(r.values())

def _check_subsets(dataset_class, subset):
    if subset not in dataset_class.subsets:
        raise ValueError(f"Invalid subset name for {dataset_class.__name__}.")

# Artificial datasets ##############################################################################

_max_int32 = 2 ** 31 - 1


class WhiteNoiseDataset(Dataset):
    subsets = []

    def __init__(self, distribution='normal', example_shape=(32, 32), size=1000, seed=53):
        self._shape = example_shape
        self._rand = np.random.RandomState(seed=seed)
        self._seeds = self._rand.random_integers(_max_int32, size=(size))
        if distribution not in ('normal', 'uniform'):
            raise ValueError('Distribution not in {"normal", "uniform"}')
        self._distribution = distribution
        super().__init__(name=f'WhiteNoise-{distribution}({example_shape})', subset=f'{seed}{size}',
                         data=self._seeds)

    def get_example(self, idx):
        self._rand.seed(self._seeds[idx])
        if self._distribution == 'normal':
            return _make_example(x=self._rand.randn(*self._shape))
        elif self._distribution == 'uniform':
            d = 12 ** 0.5 / 2
            return _make_example(x=self._rand.uniform(-d, d, self._shape))


class RademacherNoiseDataset(Dataset):
    subsets = []

    def __init__(self, example_shape=(32, 32), size=1000, seed=53):
        # lambda: np.random.binomial(n=1, p=0.5, size=(ood_num_examples, 3, 32, 32)) * 2 - 1
        self._shape = example_shape
        self._rand = np.random.RandomState(seed=seed)
        self._seeds = self._rand.random_integers(_max_int32, size=(size))
        super().__init__(name=f'RademacherNoise{example_shape}', subset=f'{seed}-{size}',
                         data=self._seeds)

        def get_example(self, idx):
            self._rand.seed(self._seeds[idx])
            return _make_example(x=self._rand.binomial(n=1, p=0.5, size=self._shape))


class HBlobsDataset(Dataset):
    subsets = []

    def __init__(self, sigma=None, example_shape=(32, 32), size=1000, seed=53):
        # lambda: np.random.binomial(n=1, p=0.5, size=(ood_num_examples, 3, 32, 32)) * 2 - 1
        self._shape = example_shape
        self._rand = np.random.RandomState(seed=seed)
        self._seeds = self._rand.random_integers(_max_int32, size=(size))
        self._sigma = sigma or 1.5 * example_shape[0] / 32
        super().__init__(name=f'HBlobs({example_shape})', subset=f'{seed}-{size}', data=self._seeds)

    def get_example(self, idx):
        self._rand.seed(self._seeds[idx])
        x = self._rand.binomial(n=1, p=0.7, size=self._shape)
        x = gblur(np.float32(x), sigma=self._sigma, multichannel=False)
        x[x < 0.75] = 0
        return _make_example(x=x)


# Classification ###################################################################################

class SVHNDataset(Dataset):
    subsets = ['trainval', 'test']

    def __init__(self, data_dir, subset='trainval'):
        _check_subsets(self.__class__, subset)
        ss = 'train' if subset == 'trainval' else subset
        data = loadmat(ss + '_32x32.mat')
        self.x, self.y = data['X'], np.remainder(data['y'], 10)
        super().__init__(subset=ss, info=dict(class_count=10, problem_id='clf'))

    def get_example(self, idx):
        return _make_example(x=self.x[idx], y=self.y[idx])

    def __len__(self):
        return len(self.x)


class Cifar10Dataset(Dataset):
    subsets = ['trainval', 'test']

    def __init__(self, data_dir, subset='trainval'):
        _check_subsets(self.__class__, subset)
        data_dir = Path(data_dir)

        ss = 'train' if subset == 'trainval' else subset

        def unpickle(file):
            with open(file, 'rb') as f:
                return pickle.load(f, encoding='latin1')

        h, w, ch = 32, 32, 3
        if ss == 'train':
            train_x = np.ndarray((0, h * w * ch), dtype=np.float32)
            train_y = []
            for i in range(1, 6):
                ds = unpickle(data_dir / f'data_batch_{i}')
                train_x = np.vstack((train_x, ds['data']))
                train_y += ds['labels']
            train_x = train_x.reshape((-1, ch, h, w)).transpose(0, 2, 3, 1)
            train_y = np.array(train_y, dtype=np.int32)
            self.x, self.y = train_x, train_y
        elif ss == 'test':
            ds = unpickle(data_dir / 'test_batch')
            test_x = ds['data'].reshape((-1, ch, h, w)).transpose(0, 2, 3, 1).astype(np.float32)
            test_y = np.array(ds['labels'], dtype=np.int32)
            self.x, self.y = test_x, test_y
        else:
            raise ValueError("The value of subset must be in {'train','test'}.")
        super().__init__(subset=ss, info=dict(class_count=10, problem_id='clf'))

    def get_example(self, idx):
        return _make_example(x=self.x[idx], y=self.y[idx])

    def __len__(self):
        return len(self.x)


class Cifar100Dataset(Dataset):
    subsets = ['trainval', 'test']

    def __init__(self, data_dir, subset='trainval'):
        _check_subsets(self.__class__, subset)

        ss = 'train' if subset == 'trainval' else subset

        def unpickle(file):
            with open(file, 'rb') as f:
                return pickle.load(f, encoding='latin1')

        data = unpickle(f"{data_dir}/{ss}")

        h, w, ch = 32, 32, 3
        train_x = data['data'].reshape((-1, ch, h, w)).transpose(0, 2, 3, 1)
        self.x, self.y = train_x, data['fine_labels']

        super().__init__(subset=ss,
                         info=dict(class_count=100, problem_id='clf',
                                   coarse_labels=data['coarse_labels']))

    def get_example(self, idx):
        return _make_example(x=self.x[idx], y=self.y[idx])

    def __len__(self):
        return len(self.x)


class DescribableTexturesDataset(Dataset):
    subsets = ['trainval', 'test']

    def __init__(self, data_dir, subset='trainval'):
        _check_subsets(self.__class__, subset)
        ss = 'train' if subset == 'trainval' else subset
        raise NotImplementedError("DescribableTexturesDataset not implemented")
        for _ in range(10):
            print("WARNING: DescribableTexturesDataset not completely implemented.")
        super().__init__(subset=subset, info=dict(class_count=47),
                         data=dset.ImageFolder(f"{data_dir}/images"), )
        self.data = dset.ImageFolder(f"{data_dir}/images")

    def get_example(self, idx):
        x, y = self.data[idx]
        return _make_example(x=np.array(x), y=y)

    def __len__(self):
        return len(self.data)


class MozgaloRVCDataset(Dataset):
    subsets = ['trainval']

    def __init__(self, data_dir, subset='trainval', remove_bottom_proportion=0.0,
                 downsampling_factor=1):
        _check_subsets(self.__class__, subset)

        ss = 'train' if subset == 'trainval' else subset
        data_dir = Path(data_dir)

        self._shape = [1104, 600]
        self._remove_bottom_proportion = remove_bottom_proportion
        self._downsampling_factor = downsampling_factor
        train_dir = data_dir / 'train'
        self._subset_dir = data_dir / ss
        subset_class_names = sorted([p.name for p in self._subset_dir.iterdir()])
        self._image_list = [p.relative_to(self._subset_dir) for p in self._subset_dir.glob('/*/*')]

        class_names = sorted([p.name for p in train_dir.iterdir()])

        modifiers = []
        if remove_bottom_proportion:
            modifiers += [f"remove_bottom_{remove_bottom_proportion:0.2f}"]
        if downsampling_factor > 1:
            modifiers += [f"downsample_{downsampling_factor}x"]
        super().__init__(subset=subset, modifiers=modifiers,
                         info=dict(class_count=25, class_names=class_names, problem_id='clf'))

    def get_example(self, idx):
        example_name = self._image_list[idx]

        def load_image():
            img = np.array(_load_image(f"{self._subset_dir}/{example_name}"))
            img = pad_to_shape(crop(img, self._shape), self._shape)
            if self._remove_bottom_proportion > 0:
                a = int(img.shape[0] * (1 - self._remove_bottom_proportion))
                img = img[:a, :, :]
            if self._downsampling_factor > 1:
                h, w = (round(x / self._downsampling_factor) for x in img.shape[:2])
                img = resize(img, (h, w))  # , anti_aliasing=True)
            return img

        lab_str = example_name.parent
        return _make_example(x_=load_image, y=self.info['class_names'].index(lab_str))

    def __len__(self):
        return len(self._image_list)


class TinyImageNetDataset(Dataset):
    subsets = ['train', 'val', 'test']

    def __init__(self, data_dir, subset='trainval'):
        _check_subsets(self.__class__, subset)
        data_dir = Path(data_dir)

        with open(data_dir / "wnids.txt") as fs:
            class_names = [l.strip() for l in fs.readlines()]
        subset_dir = data_dir / subset

        self._examples = []

        if subset == 'train':
            for i, class_name in enumerate(class_names):
                images_dir = subset_dir / class_name / "images"
                for im in images_dir.iterdir():
                    self._examples.append((images_dir / im, i))
        elif subset == 'val':
            with open(subset_dir / "val_annotations.txt") as fs:
                im_labs = [l.split()[:2] for l in fs.readlines()]
                images_dir = subset_dir / "images"
                for im, lab in im_labs:
                    lab = class_names.index(lab)
                    self._examples.append((images_dir / im, lab))
        elif subset == 'test':
            images_dir = subset_dir / "images"
            self._examples = [(images_dir / im, -1) for im in images_dir.iterdir()]

        self.name = f"TinyImageNet-{subset}"
        super().__init__(subset=subset,
                         info=dict(class_count=200, class_names=class_names, problem_id='clf'))

    def get_example(self, idx):
        img_path, lab = self._examples[idx]
        return _make_example(x_=lambda: np.array(_load_image(img_path)), y=lab)

    def __len__(self):
        return len(self._examples)


class INaturalist2018Dataset(Dataset):
    subsets = ['train', 'val', 'test']

    def __init__(self, data_dir, subset='train', superspecies='all'):
        _check_subsets(self.__class__, subset)
        data_dir = Path(data_dir)
        self._data_dir = data_dir

        with open(f"{data_dir}/{subset}2018.json") as fs:
            info = json.loads(fs.read())
        self._file_names = [x['file_name'] for x in info['images']]
        if 'annotations' in info.keys():
            self._labels = [x['category_id'] for x in info['annotations']]
        else:
            self._labels = np.full(shape=len(self._file_names), fill_value=-1)

        with open(data_dir / "categories.json") as fs:
            categories = json.loads(fs.read())

        super().__init__(subset=subset,
                         info=dict(class_count=8142, class_to_categories=categories,
                                   problem_id='clf'))

    def get_example(self, idx):
        def load_img():
            img_path = self._data_dir / self._file_names[idx]
            img = pimg.open(img_path).convert('RGB')
            img = img.resize(np.array(img.size) // 4, pimg.BILINEAR)
            return pad_to_shape(np.array(img), [200, 200])

        return _make_example(x_=load_img, y=self._labels[idx])

    def __len__(self):
        return len(self._labels)


class TinyImagesDataset(Dataset):
    # Taken (and slightly modified) from
    # https://github.com/hendrycks/outlier-exposure/blob/master/utils/tinyimages_80mn_loader.py
    subsets = []

    def __init__(self, data_dir, exclude_cifar=False, cifar_indexes_file=None):

        def load_image(idx):
            with open(f'{data_dir}/tiny_images.bin', "rb") as data_file:
                data_file.seek(idx * 3072)
                data = data_file.read(3072)
                return np.fromstring(data, dtype='uint8').reshape(32, 32, 3, order="F")

        self.load_image = load_image

        self.exclude_cifar = exclude_cifar

        if exclude_cifar:
            from bisect import bisect_left
            self.cifar_idxs = []
            with open(cifar_indexes_file, 'r') as idxs:
                for idx in idxs:
                    # indices in file take the 80mn database to start at 1, hence "- 1"
                    self.cifar_idxs.append(int(idx) - 1)
            self.cifar_idxs = tuple(sorted(self.cifar_idxs))

            def binary_search(x, hi=len(self.cifar_idxs)):
                pos = bisect_left(self.cifar_idxs, x, 0, hi)  # find insertion position
                return True if pos != hi and self.cifar_idxs[pos] == x else False

            self.in_cifar = binary_search
        super().__init__(info=dict(id='tinyimages', problem_id='clf'))  # TODO: class_count

    def get_example(self, idx):
        if self.exclude_cifar:
            while self.in_cifar(idx):
                idx = np.random.randint(79302017)
        return _make_example(lambda: self.load_image(idx), y=-1)

    def __len__(self):
        return 79302017


# Semantic segmentation ############################################################################

class CamVidDataset(Dataset):
    # https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid
    subsets = ['train', 'val', 'test']

    def __init__(self, data_dir, subset='train'):
        _check_subsets(self.__class__, subset)

        lines = Path(f'{data_dir}/{subset}.txt').read_text().splitlines()
        self._img_lab_list = [
            [f"{data_dir}/{p.replace('/SegNet/CamVid/', '')}" for p in line.split()]
            for line in lines]

        info = {
            'problem_id':
                'semseg',
            'class_count':
                11,
            'class_names': [
                'Sky', 'Building', 'Pole', 'Road', 'Pavement', 'Tree',
                'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist'
            ],
            'class_colors': [
                (128, 128, 128),
                (128, 0, 0),
                (192, 192, 128),
                (128, 64, 128),
                (60, 40, 222),
                (128, 128, 0),
                (192, 128, 128),
                (64, 64, 128),
                (64, 0, 128),
                (64, 64, 0),
                (0, 128, 192),
            ]
        }
        super().__init__(subset=subset, info=info)

    def get_example(self, idx):
        ip, lp = self._img_lab_list[idx]

        def load_label():
            lab = np.array(_load_image(lp, force_rgb=False).astype(np.int8))
            lab[lab == 11] = -1
            return lab

        return _make_example(x_=lambda: np.array(_load_image(ip)), y_=load_label)

    def __len__(self):
        return len(self._img_lab_list)


class CityscapesFineDataset(Dataset):
    subsets = ['train', 'val', 'test']  # 'test' labels are invalid

    def __init__(self, data_dir, subset='train', downsampling_factor=1, remove_hood=False):
        _check_subsets(self.__class__, subset)
        if downsampling_factor <= 1:
            raise ValueError("downsampling_factor must be greater or equal to 1.")

        self._downsampling_factor = downsampling_factor
        self._shape = np.array([1024, 2048]) // downsampling_factor
        self._remove_hood = remove_hood

        IMG_SUFFIX = "_leftImg8bit.png"
        LAB_SUFFIX = "_gtFine_labelIds.png"
        self._id_to_label = [(l.id, l.trainId) for l in cslabels]

        self._images_dir = Path(f'{data_dir}/left/leftImg8bit/{subset}')
        self._labels_dir = Path(f'{data_dir}/fine_annotations/{subset}')
        self._image_list = [x.relative_to(self._images_dir) for x in self._images_dir.glob('/*/*')]
        self._label_list = [x[:-len(IMG_SUFFIX)] + LAB_SUFFIX for x in self._image_list]

        info = {
            'problem_id': 'semseg',
            'class_count': 19,
            'class_names': [l.name for l in cslabels if l.trainId >= 0],
            'class_colors': [l.color for l in cslabels if l.trainId >= 0],
        }
        modifiers = []
        if downsampling_factor > 1:
            modifiers += f"downsample({downsampling_factor})"
        if remove_hood:
            modifiers += f"remove_hood"
        super().__init__(subset=subset, modifiers=modifiers, info=info)

    def get_example(self, idx):
        rh_height = self._shape[0] * 7 // 8

        def load_image():
            img = pimg.open(self._images_dir / self._image_list[idx])
            if self._downsampling_factor > 1:
                img = img.resize(self._shape[::-1], pimg.BILINEAR)
            img = np.array(img, dtype=np.uint8)
            if self._remove_hood:
                img = img[:rh_height, :, :]
            return img

        def load_label():
            lab = pimg.open(self._labels_dir / self._label_list[idx])
            if self._downsampling_factor > 1:
                lab = lab.resize(self._shape[::-1], pimg.NEAREST)
            lab = np.array(lab, dtype=np.int8)
            for id, lb in self._id_to_label:
                lab[lab == id] = lb
            if self._remove_hood:
                lab = lab[:rh_height, :]
            return lab

        return _make_example(x_=load_image, y_=load_label)

    def __len__(self):
        return len(self._image_list)


class WildDashDataset(Dataset):
    subsets = ['val', 'bench', 'both']
    splits = dict(all=(('val', 'bench'), None), both=(('val', 'bench'), None))

    def __init__(self, data_dir, subset='val', downsampling_factor=1):
        _check_subsets(self.__class__, subset)
        if downsampling_factor <= 1:
            raise ValueError("downsampling_factor must be greater or equal to 1.")

        self._subset = subset

        self._downsampling_factor = downsampling_factor
        self._shape = np.array([1070, 1920]) // downsampling_factor

        self._IMG_SUFFIX = "0.png"
        self._LAB_SUFFIX = "0_labelIds.png"
        self._id_to_label = [(l.id, l.trainId) for l in cslabels]

        self._images_dir = Path(f'{data_dir}/wd_{subset}_01')
        self._image_names = sorted([
            str(x.relative_to(self._images_dir))[:-5]
            for x in self._images_dir.glob(f'/*{self._IMG_SUFFIX}')
        ])
        info = {
            'problem_id': 'semseg',
            'class_count': 19,
            'class_names': [l.name for l in cslabels if l.trainId >= 0],
            'class_colors': [l.color for l in cslabels if l.trainId >= 0],
        }
        self._blank_label = np.full(list(self._shape), -1, dtype=np.int8)
        modifiers = [f"downsample({downsampling_factor})"] if downsampling_factor > 1 else []
        super().__init__(subset=subset, modifiers=modifiers, info=info)

    def get_example(self, idx):
        path_prefix = f"{self._images_dir}/{self._image_names[idx]}"

        def load_img():
            img = pimg.open(f"{path_prefix}{self._IMG_SUFFIX}").convert('RGB')
            if self._downsampling_factor > 1:
                img = img.resize(self._shape[::-1], pimg.BILINEAR)
            return np.array(img, dtype=np.uint8)

        def load_lab():
            if self._subset == 'bench':
                lab = self._blank_label
            else:
                lab = pimg.open(f"{path_prefix}{self._LAB_SUFFIX}")
                if self._downsampling_factor > 1:
                    lab = lab.resize(self._shape[::-1], pimg.NEAREST)
                lab = np.array(lab, dtype=np.int8)
            for id, lb in self._id_to_label:
                lab[lab == id] = lb
            return lab

        return _make_example(x_=load_img, y_=load_lab)

    def __len__(self):
        return len(self._image_names)


class ICCV09Dataset(Dataset):
    subsets = []

    def __init__(self, data_dir):  # TODO subset
        self._shape = [240, 320]
        self._images_dir = Path(f'{data_dir}/images')
        self._labels_dir = Path(f'{data_dir}/labels')
        self._image_list = [str(x)[:-4] for x in self._images_dir.iterdir()]

        info = {
            'problem_id': 'semseg',
            'class_count': 8,
            'class_names': ['sky', 'tree', 'road', 'grass', 'water', 'building', 'mountain',
                            'foreground object']
        }
        super().__init__(info=info)

    def get_example(self, idx):
        name = self._image_list[idx]

        def load_img():
            img = np.array(_load_image(self._images_dir / f"{name}.jpg"))
            return pad_to_shape(crop(img, self._shape), self._shape)

        def load_lab():
            lab = np.loadtxt(self._labels_dir / f"{name}.regions.txt", dtype=np.int8)
            return pad_to_shape(crop(lab, self._shape), self._shape, value=-1)

        return _make_example(x_=load_img, y_=load_lab)

    def __len__(self):
        return len(self._image_list)


class VOC2012SegmentationDataset(Dataset):
    subsets = ['train', 'val', 'trainval', 'test']

    def __init__(self, data_dir, subset='train'):
        _check_subsets(self.__class__, subset)
        data_dir = Path(data_dir)

        sets_dir = data_dir / 'ImageSets/Segmentation'
        self._images_dir = data_dir / 'JPEGImages'
        self._labels_dir = data_dir / 'SegmentationClass'
        self._image_list = (sets_dir / f'{subset}.txt').read_text().splitlines()
        info = {
            'problem_id': 'semseg',
            'class_count': 21,
            'class_names': ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                            'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                            'train', 'tvmonitor'],
            'class_colors': [
                (128, 64, 128),
                (244, 35, 232),
                (70, 70, 70),
                (102, 102, 156),
                (190, 153, 153),
                (153, 153, 153),
                (250, 170, 30),
                (220, 220, 0),
                (107, 142, 35),
                (152, 251, 152),
                (70, 130, 180),
                (220, 20, 60),
                (255, 0, 0),
                (0, 0, 142),
                (0, 0, 70),
                (0, 60, 100),
                (0, 80, 100),
                (0, 0, 230),
                (0, 0, 230),
                (0, 0, 230),
                (119, 11, 32),
            ]
        }
        super().__init__(subset=subset, info=info)

    def get_example(self, idx):
        name = self._image_list[idx]

        def load_img():
            img = np.array(_load_image(self._images_dir / f"{name}.jpg"))
            return pad_to_shape(img, [500] * 2)

        def load_lab():
            lab = np.array(
                _load_image(self._labels_dir / f"{name}.png", force_rgb=False).astype(np.int8))
            return pad_to_shape(lab, [500] * 2, value=-1)  # -1 ok?

        return _make_example(x_=load_img, y_=load_lab)

    def __len__(self):
        return len(self._image_list)


# Other

class ISUNDataset(Dataset):
    # https://github.com/matthias-k/pysaliency/blob/master/pysaliency/external_datasets.py
    # TODO: labels, problem_id
    subsets = ['train', 'val', 'test']

    def __init__(self, data_dir, subset='train'):
        _check_subsets(self.__class__, subset)
        self._images_dir = f'{data_dir}/images'
        subset = {'train': 'training', 'val': 'validation', 'test': 'testing'}[subset]

        data_file = f'{data_dir}/{subset}.mat'
        data = loadmat(data_file)[subset]
        self._image_names = [d[0] for d in data['image'][:, 0]]

        super().__init__(subset=subset, info=dict(problem_id=None))

    def get_example(self, idx):
        return _make_example(
            x_=lambda: np.array(
                _load_image(f"{self._images_dir}/{self._image_names[idx]}.jpg")),
            y=-1)

    def __len__(self):
        return len(self._image_names)


class LSUNDataset(Dataset):
    # TODO: labels, replace with LSUNDatasetNew
    subsets = ['test']

    def __init__(self, data_dir, subset='train'):
        _check_subsets(self.__class__, subset)

        self._subset_dir = f'{data_dir}/{subset}'
        self._image_names = [
            os.path.relpath(x, start=self._subset_dir)
            for x in glob.glob(f'{self._subset_dir}/**/*.webp', recursive=True)
        ]
        self.info = dict(id='LSUN', problem_id=None)
        self.name = f"LSUN-{subset}"

    def get_example(self, idx):
        return _make_example(
            x_=lambda: np.array(_load_image(f"{self._subset_dir}/{self._image_names[idx]}")),
            y=-1)

    def __len__(self):
        return len(self._image_names)


class _LSUNClass(Dataset):
    def __init__(self, db_path):
        import lmdb
        self.db_path = db_path
        self.env = lmdb.open(db_path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        cache_file = Path('_cache_' + db_path.replace('/', '_'))
        if cache_file.is_file():
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key, _ in txn.cursor()]
            pickle.dump(self.keys, open(cache_file, "wb"))

    def get_example(self, index):
        img, target = None, None  # TODO: target
        env = self.env
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])

        buf = BytesIO()
        buf.write(imgbuf)
        buf.seek(0)

        return _make_example(x_=lambda: np.array(pimg.open(buf).convert('RGB')), y_=target)

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


class LSUNDatasetNew(Dataset):
    """
    `LSUN <http://lsun.cs.princeton.edu>`_ dataset.
    Args:
        db_path (string): Root directory for the database files.
        classes (string or list): One of {'train', 'val', 'test'} or a list of
            categories to load. e,g. ['bedroom_train', 'church_train'].
    """
    subsets = ['train', 'val', 'test']

    def __init__(self, db_path, subset, categories=None):
        _check_subsets(self.__class__, subset)
        categories = categories or ['bedroom', 'bridge', 'church_outdoor', 'classroom',
                                    'conference_room', 'dining_room', 'kitchen',
                                    'living_room', 'restaurant', 'tower']
        self.db_path = db_path
        if subset == 'test':
            self.classes = [subset]
        else:
            self.classes = [c + '_' + subset for c in categories]

        # for each class, create an LSUNClassDataset
        self.dbs = []
        for c in self.classes:
            self.dbs.append(_LSUNClass(
                db_path=db_path + '/' + c + '_lmdb'))

        self.indices = []
        self.length = 0
        for db in self.dbs:
            self.length += len(db)
            self.indices.append(self.length)

    def get_example(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target) where target is the index of the target category.
        """
        target = 0
        sub = 0
        for ind in self.indices:
            if index < ind:
                break
            target += 1
            sub = ind

        db = self.dbs[target]
        index = index - sub
        return _make_example(x=lambda: db[index][0], y=target)

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'
