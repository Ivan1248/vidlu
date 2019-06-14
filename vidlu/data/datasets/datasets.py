import pickle
import json
import tarfile
import gzip
import zipfile
from pathlib import Path
import shutil
import warnings

import PIL.Image as pimg
import numpy as np
from scipy.io import loadmat
import torchvision.datasets as dset
import torchvision.transforms.functional as tvtf
from tqdm import tqdm

from .. import Dataset, Record
from vidlu.utils.misc import download
from vidlu.transforms import numpy as numpy_transforms

from ._cityscapes_labels import labels as cslabels


# Helper functions


def _load_image(path, force_rgb=True):
    img = pimg.open(path)
    if force_rgb and img.mode != 'RGB':
        img = img.convert('BGR')
    return img


def _rescale(img, factor, interpolation=pimg.BILINEAR):
    return img.resize([round(d * factor) for d in img.size], interpolation)


def _make_record(**kwargs):
    def array_to_image(k, v):
        if (k == 'x' and isinstance(v, np.ndarray) and v.dtype == np.uint8 and
                2 <= len(v.shape) <= 3):
            return pimg.fromarray(v)  # automatic RGB or L, depending on shape
        return v

    return Record(**{k: array_to_image(k, v) for k, v in kwargs.items()})


def _check_subsets(dataset_class, subset):
    if subset not in dataset_class.subsets:
        raise ValueError(f"Invalid subset name for {dataset_class.__name__}.")


def load_image_with_downsampling(path, downsampling):
    if not isinstance(downsampling, int):
        raise ValueError("`downsampling` must be an `int`.")
    img = _load_image(path)
    if downsampling > 1:
        img = tvtf.resize(img, np.flip(img.size) // downsampling, pimg.BILINEAR)
    return img


def load_segmentation_with_downsampling(path, downsampling, id_to_label=dict(),
                                        dtype=np.int8):
    """ Loads and optionally translates segmentation labels.

    Args:
        path: (path-like) label file path.
        downsampling: (int) an integer larger than 1.
        id_to_label: (dict, optional) a dictionary for translating labels. Keys
            can be tuples (if colors are translated to integers) or integers.
        dtype: output element type.

    Returns:
        A 2D array.
    """
    if not isinstance(downsampling, int):
        raise ValueError("`downsampling` must be an `int`.")

    lab = _load_image(path, force_rgb=False)
    if downsampling > 1:
        lab = tvtf.resize(lab, np.flip(lab.size) // downsampling, pimg.NEAREST)

    if len(lab.getbands()) != 1:  # for rgb labels
        lab = np.array(lab)
        scalarizer = np.array([256 ** 2, 256, 1])
        u, inv = np.unique(lab.reshape(-1, 3).dot(scalarizer), return_inverse=True)
        id_to_label = {np.array(k).dot(scalarizer): v for k, v in id_to_label.items()}
        return np.array([id_to_label.get(k, -1)
                         for k in u], dtype=dtype)[inv].reshape(lab.shape[:2])
    elif len(id_to_label) > 140:  # faster for great numbers of distinct labels
        lab = np.array(lab)
        u, inv = np.unique(lab, return_inverse=True)
        return np.array([id_to_label.get(k, k) for k in u], dtype=dtype)[inv].reshape(lab.shape)
    else:  # faster for small numbers of distinct labels
        lab = np.array(lab, dtype=dtype)
        for id, lb in id_to_label.items():
            lab[lab == id] = lb
        return lab


# Artificial datasets ##############################################################################

_max_int32 = 2 ** 31 - 1


class WhiteNoise(Dataset):
    subsets = []

    def __init__(self, distribution='normal', example_shape=(32, 32, 3), size=1000, seed=53):
        self._shape = example_shape
        self._rand = np.random.RandomState(seed=seed)
        self._seeds = self._rand.random_integers(_max_int32, size=(size,))
        if distribution not in ('normal', 'uniform'):
            raise ValueError('Distribution not in {"normal", "uniform"}')
        self._distribution = distribution
        super().__init__(name=f'WhiteNoise-{distribution}({example_shape})', subset=f'{seed}{size}',
                         data=self._seeds)

    def get_example(self, idx):
        self._rand.seed(self._seeds[idx])
        if self._distribution == 'normal':
            return _make_record(x=self._rand.randn(*self._shape))
        elif self._distribution == 'uniform':
            d = 12 ** 0.5 / 2
            return _make_record(x=self._rand.uniform(-d, d, self._shape))


class RademacherNoise(Dataset):
    subsets = []

    def __init__(self, example_shape=(32, 32, 3), size=1000, seed=53):
        # lambda: np.random.binomial(n=1, p=0.5, size=(ood_num_examples, 3, 32, 32)) * 2 - 1
        self._shape = example_shape
        self._rand = np.random.RandomState(seed=seed)
        self._seeds = self._rand.random_integers(_max_int32, size=(size,))
        super().__init__(
            name=f'RademacherNoise{example_shape}',
            subset=f'{seed}-{size}',
            data=self._seeds)

        def get_example(self, idx):
            self._rand.seed(self._seeds[idx])
            return _make_record(
                x=self._rand.binomial(n=1, p=0.5, size=self._shape))


class HBlobs(Dataset):
    subsets = []

    def __init__(self, sigma=None, example_shape=(32, 32, 3), size=1000, seed=53):
        # lambda: np.random.binomial(n=1, p=0.5, size=(ood_num_examples, 3, 32, 32)) * 2 - 1
        self._shape = example_shape
        self._rand = np.random.RandomState(seed=seed)
        self._seeds = self._rand.random_integers(_max_int32, size=(size,))
        self._sigma = sigma or 1.5 * example_shape[0] / 32
        super().__init__(name=f'HBlobs({example_shape})', subset=f'{seed}-{size}', data=self._seeds)

    def get_example(self, idx):
        from skimage.filters import gaussian
        self._rand.seed(self._seeds[idx])
        x = self._rand.binomial(n=1, p=0.7, size=self._shape)
        x = gaussian(np.float32(x), sigma=self._sigma, multichannel=False)
        x[x < 0.75] = 0
        return _make_record(x=x)


class DummyClassification(Dataset):
    subsets = []

    def __init__(self, example_shape=(32, 32, 3), size=1000):
        self._shape = example_shape
        self._colors = [row for row in np.eye(3, 3) * 255]
        self._len = size
        super().__init__(name=f'DummyClassification({example_shape})', subset=f'{size}',
                         info=dict(class_count=len(self._colors), problem='classification'))

    def get_example(self, idx):
        color_idx = idx % len(self._colors)
        return _make_record(x=np.ones(self._shape) * self._colors[color_idx], y=color_idx)

    def __len__(self):
        return self._len


# Classification ###################################################################################


class MNIST(Dataset):
    subsets = ['trainval', 'test']
    default_dir = 'MNIST'
    _files = dict(x_train='train-images-idx3-ubyte', y_train='train-labels-idx1-ubyte',
                  x_test='t10k-images-idx3-ubyte', y_test='t10k-labels-idx1-ubyte')

    def __init__(self, data_dir, subset='trainval'):
        _check_subsets(self.__class__, subset)
        data_dir = Path(data_dir)

        self.download_if_necessary(data_dir)

        x_path = data_dir / self._files['x_test' if subset == 'test' else 'x_train']
        y_path = data_dir / self._files['y_test' if subset == 'test' else 'y_train']
        self.x, self.y = self.load_array(x_path, x=True), self.load_array(y_path, x=False)
        super().__init__(subset=subset, info=dict(class_count=10, problem='classification'))

    def download(self, data_dir):
        url_base = 'http://yann.lecun.com/exdb/mnist/'
        print(f"Downloading dataset to {data_dir}")
        data_dir.mkdir(exist_ok=True)
        for p in type(self)._files.values():
            final_path = data_dir / p
            download_path = final_path.with_suffix('.gz')
            download(url=url_base + p + '.gz', output_path=download_path)
            with gzip.open(download_path, 'rb') as gz:
                with open(final_path, 'wb') as raw:
                    raw.write(gz.read())
            download_path.unlink()

    @staticmethod
    def load_array(path, x):
        with open(path, 'rb') as f:
            return (np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28) if x
                    else np.frombuffer(f.read(), np.uint8, offset=8))

    def get_example(self, idx):
        return _make_record(x=self.x[idx], y=self.y[idx])

    def __len__(self):
        return len(self.y)


class SVHNDataset(Dataset):
    subsets = ['trainval', 'test']
    default_dir = 'SVHN'

    def __init__(self, data_dir, subset='trainval'):
        _check_subsets(self.__class__, subset)
        ss = 'train' if subset == 'trainval' else subset
        data = loadmat(Path(data_dir) / (ss + '_32x32.mat'))
        self.x, self.y = data['X'], np.remainder(data['y'], 10)
        super().__init__(subset=subset, info=dict(class_count=10, problem='classification'))

    def get_example(self, idx):
        return _make_record(x=self.x[idx], y=self.y[idx])

    def __len__(self):
        return len(self.x)


class Cifar10(Dataset):
    subsets = ['trainval', 'test']
    default_dir = 'cifar-10-batches-py'

    def download(self, data_dir):
        datasets_dir = data_dir.parent
        download_path = datasets_dir / "cifar-10-python.tar.gz"
        print(f"Downloading dataset to {datasets_dir}")
        download(url="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
                 output_path=download_path, md5='c58f30108f718f92721af3b95e74349a')
        with tarfile.open(download_path, "r:gz") as tar:
            tar.extractall(path=datasets_dir)
        download_path.unlink()

    def __init__(self, data_dir, subset='trainval'):
        _check_subsets(self.__class__, subset)
        data_dir = Path(data_dir)

        self.download_if_necessary(data_dir)

        ss = 'train' if subset == 'trainval' else subset

        def unpickle(file):
            with open(file, 'rb') as f:
                return pickle.load(f, encoding='latin1')

        h, w, ch = 32, 32, 3
        if ss == 'train':
            train_x = np.ndarray((0, h * w * ch), dtype=np.uint8)
            train_y = []
            for i in range(1, 6):
                ds = unpickle(data_dir / f'data_batch_{i}')
                train_x = np.vstack((train_x, ds['data']))
                train_y += ds['labels']
            train_x = train_x.reshape((-1, ch, h, w)).transpose(0, 2, 3, 1)
            train_y = np.array(train_y, dtype=np.int8)
            self.x, self.y = train_x, train_y
        elif ss == 'test':
            ds = unpickle(data_dir / 'test_batch')
            test_x = ds['data'].reshape((-1, ch, h, w)).transpose(0, 2, 3, 1)
            test_y = np.array(ds['labels'], dtype=np.int8)
            self.x, self.y = test_x, test_y
        else:
            raise ValueError("The value of subset must be in {'train','test'}.")
        super().__init__(subset=subset, info=dict(class_count=10, problem='classification'))

    def get_example(self, idx):
        return _make_record(x=self.x[idx], y=self.y[idx])

    def __len__(self):
        return len(self.x)


class Cifar100(Dataset):
    subsets = ['trainval', 'test']
    default_dir = 'cifar-100-python'

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

        super().__init__(subset=subset, info=dict(class_count=100, problem='classification',
                                                  coarse_labels=data['coarse_labels']))

    def get_example(self, idx):
        return _make_record(x=self.x[idx], y=self.y[idx])

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
                         data=dset.ImageFolder(f"{data_dir}/images"))
        self.data = dset.ImageFolder(f"{data_dir}/images")

    def get_example(self, idx):
        x, y = self.data[idx]
        return _make_record(x=x, y=y)

    def __len__(self):
        return len(self.data)


class TinyImageNet(Dataset):
    subsets = ['train', 'val', 'test']
    default_dir = 'tiny-imagenet-200'

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
        super().__init__(subset=subset, info=dict(class_count=200, class_names=class_names,
                                                  problem='classification'))

    def get_example(self, idx):
        img_path, lab = self._examples[idx]
        return _make_record(x_=lambda: _load_image(img_path), y=lab)

    def __len__(self):
        return len(self._examples)


class INaturalist2018(Dataset):
    subsets = 'train', 'val', 'test'
    default_dir = 'iNaturalist2018'

    url = "https://github.com/visipedia/inat_comp"
    categories = ("http://www.vision.caltech.edu/~gvanhorn/datasets/" +
                  "inaturalist/fgvc5_competition/categories.json.tar.gz")

    def __init__(self, data_dir, subset='train', superspecies='all', downsampling=1):
        _check_subsets(self.__class__, subset)
        data_dir = Path(data_dir)
        self._data_dir = data_dir

        self._downsampling = downsampling

        with open(f"{data_dir}/{subset}2018.json") as fs:
            info = json.loads(fs.read())
        self._file_names = [x['file_name'] for x in info['images']]
        if 'annotations' in info.keys():
            self._labels = [x['category_id'] for x in info['annotations']]
        else:
            self._labels = np.full(shape=len(self._file_names), fill_value=-1)

        info = dict(class_count=8142, problem='classification')
        categories_path = data_dir / "categories.json"
        if categories_path.exists():
            with open(categories_path) as fs:
                info['class_to_categories'] = json.loads(fs.read())
        else:
            warnings.warn(f"categories.json containing category names is missing from {data_dir}."
                          + f" It can be obtained from {INaturalist2018.categories}")

        super().__init__(subset=subset, info=info)

    def get_example(self, idx):
        def load_img():
            img_path = self._data_dir / self._file_names[idx]
            img = _load_image(img_path)
            img = _rescale(img, 1 / self._downsampling)
            return tvtf.center_crop(img, [800 * self._downsampling] * 2)

        return _make_record(x_=load_img, y=self._labels[idx])

    def __len__(self):
        return len(self._labels)


class TinyImages(Dataset):
    # Taken (and slightly modified) from
    # https://github.com/hendrycks/outlier-exposure/blob/master/utils/tinyimages_80mn_loader.py
    subsets = []
    default_dir = 'tiny-images'

    def __init__(self, data_dir, exclude_cifar=False, cifar_indexes_file=None):
        def load_image(idx):
            with open(f'{data_dir}/tiny_images.bin', "rb") as data_file:
                data_file.seek(idx * 3072)
                data = data_file.read(3072)
                return np.fromstring(
                    data, dtype='uint8').reshape(
                    32, 32, 3, order="F")

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
        super().__init__(info=dict(
            id='tinyimages', problem='classification'))  # TODO: class_count

    def get_example(self, idx):
        if self.exclude_cifar:
            while self.in_cifar(idx):
                idx = np.random.randint(79302017)
        return _make_record(x_=lambda: self.load_image(idx), y=-1)

    def __len__(self):
        return 79302017


# Semantic segmentation ############################################################################


class CamVid(Dataset):
    subsets = ['train', 'val', 'test']
    default_dir = 'CamVid'
    class_groups_colors = {
        'Sky': {'Sky': (128, 128, 128)},
        'Buliding': {'Building': (128, 0, 0)},
        'Pole': {'Column_Pole': (192, 192, 128)},
        'Road': {'Road': (128, 64, 128), 'LaneMkgsDriv': (128, 0, 192),
                 'LaneMkgsNonDriv': (192, 0, 64), 'RoadShoulder': (128, 128, 192)},
        'Sidewalk': {'Sidewalk': (0, 0, 192)},
        'Tree': {'Tree': (128, 128, 0)},
        'SignSymbol': {'SignSymbol': (192, 128, 128)},
        'Fence': {'Fence': (64, 64, 128)},
        'Vehicle': {'Car': (64, 0, 128), 'Truck_Bus': (192, 128, 192), 'Train': (192, 64, 128),
                    'SUVPickupTruck': (64, 128, 192)},
        'Pedestrian': {'Pedestrian': (64, 64, 0), 'Child': (192, 128, 64)},
        'Bicyclist': {'Bicyclist': (0, 128, 192)},
        'Void': {'Void': (0, 0, 0)}
    }

    def download(self, data_dir):
        datasets_dir = Path(data_dir).parent
        download_path = datasets_dir / "CamVid.zip"
        download(url="https://github.com/Ivan1248/CamVid/archive/master.zip",
                 output_path=download_path)
        print(f"Extracting dataset to {datasets_dir}")
        with zipfile.ZipFile(download_path) as archive:
            for filename in tqdm(archive.namelist(), f"Extracting {download_path}"):
                archive.extract(filename, datasets_dir)
        shutil.move(datasets_dir / 'CamVid-master', data_dir)
        download_path.unlink()

    def __init__(self, data_dir, subset='train', downsampling=1):
        _check_subsets(self.__class__, subset)
        if downsampling < 1:
            raise ValueError(
                "downsampling must be greater or equal to 1.")

        data_dir = Path(data_dir)
        self.download_if_necessary(data_dir)

        self._downsampling = downsampling

        img_dir = data_dir / '701_StillsRaw_full'
        lab_dir = data_dir / 'LabeledApproved_full'
        lines = (data_dir / f'{subset}.txt').read_text().splitlines()
        self._img_lab_list = [(str(img_dir / f'{x}.png'), str(lab_dir / f'{x}_L.png'))
                              for x in lines]
        info = dict(
            problem='semantic_segmentation',
            class_count=11,
            class_names=list(CamVid.class_groups_colors.keys()),
            class_colors=[next(iter(v.values())) for v in
                          CamVid.class_groups_colors.values()])

        self.color_to_label = dict()
        for i, (k, v) in enumerate(CamVid.class_groups_colors.items()):
            for c, color in v.items():
                self.color_to_label[color] = i
        self.color_to_label[(0, 0, 0)] = -1

        super().__init__(subset=subset, info=info)

    def get_example(self, idx):
        ip, lp = self._img_lab_list[idx]
        df = self._downsampling
        return _make_record(
            x_=lambda: load_image_with_downsampling(ip, df),
            y_=lambda: load_segmentation_with_downsampling(lp, df, self.color_to_label))

    def __len__(self):
        return len(self._img_lab_list)


class Cityscapes(Dataset):
    subsets = ['train', 'val', 'test']  # 'test' labels are invalid
    default_dir = 'Cityscapes'

    def __init__(self, data_dir, subset='train', downsampling=1):
        _check_subsets(self.__class__, subset)
        if downsampling < 1:
            raise ValueError(
                "downsampling must be greater or equal to 1.")

        self._downsampling = downsampling
        self._shape = np.array([1024, 2048]) // downsampling

        IMG_SUFFIX = "_leftImg8bit.png"
        LAB_SUFFIX = "_gtFine_labelIds.png"
        self._id_to_label = {l.id: l.trainId for l in cslabels}

        self._images_dir = Path(f'{data_dir}/leftImg8bit/{subset}')
        self._labels_dir = Path(f'{data_dir}/gtFine/{subset}')
        self._image_list = [x.relative_to(self._images_dir) for x in self._images_dir.glob('*/*')]
        self._label_list = [str(x)[:-len(IMG_SUFFIX)] + LAB_SUFFIX for x in self._image_list]

        info = dict(problem='semantic_segmentation', class_count=19,
                    class_names=[l.name for l in cslabels if l.trainId >= 0],
                    class_colors=[l.color for l in cslabels if l.trainId >= 0])
        modifiers = [f"downsample({downsampling})"] if downsampling > 1 else []
        super().__init__(subset=subset, modifiers=modifiers, info=info)

    def get_example(self, idx):
        ip = self._images_dir / self._image_list[idx]
        lp = self._labels_dir / self._label_list[idx]
        df = self._downsampling
        return _make_record(
            x_=lambda: load_image_with_downsampling(ip, df),
            y_=lambda: load_segmentation_with_downsampling(lp, df, self._id_to_label))

    def __len__(self):
        return len(self._image_list)


class WildDash(Dataset):
    subsets = ['val', 'bench', 'both']
    splits = dict(all=(('val', 'bench'), None), both=(('val', 'bench'), None))
    default_dir = 'WildDash'

    def __init__(self, data_dir, subset='val', downsampling=1):
        _check_subsets(self.__class__, subset)
        if downsampling < 1:
            raise ValueError(
                "downsampling must be greater or equal to 1.")

        self._subset = subset

        self._downsampling = downsampling
        self._shape = np.array([1070, 1920]) // downsampling

        self._IMG_SUFFIX = "0.png"
        self._LAB_SUFFIX = "0_labelIds.png"
        self._id_to_label = [(l.id, l.trainId) for l in cslabels]

        self._images_dir = Path(f'{data_dir}/wd_{subset}_01')
        self._image_names = sorted([
            str(x.relative_to(self._images_dir))[:-5]
            for x in self._images_dir.glob(f'/*{self._IMG_SUFFIX}')
        ])
        info = {
            'problem': 'semantic_segmentation',
            'class_count': 19,
            'class_names': [l.name for l in cslabels if l.trainId >= 0],
            'class_colors': [l.color for l in cslabels if l.trainId >= 0],
        }
        self._blank_label = np.full(list(self._shape), -1, dtype=np.int8)
        modifiers = [f"downsample({downsampling})"
                     ] if downsampling > 1 else []
        super().__init__(subset=subset, modifiers=modifiers, info=info)

    def get_example(self, idx):
        path_prefix = f"{self._images_dir}/{self._image_names[idx]}"

        def load_img():
            img = pimg.open(f"{path_prefix}{self._IMG_SUFFIX}").convert('RGB')
            if self._downsampling > 1:
                img = tvtf.resize(img, self._shape, pimg.BILINEAR)
            return img

        def load_lab():
            if self._subset == 'bench':
                lab = self._blank_label
            else:
                lab = pimg.open(f"{path_prefix}{self._LAB_SUFFIX}")
                if self._downsampling > 1:
                    lab = tvtf.resize(lab, self._shape, pimg.NEAREST)
                lab = np.array(lab, dtype=np.int8)
            for id, lb in self._id_to_label:
                lab[lab == id] = lb
            return lab

        return _make_record(x_=load_img, y_=load_lab)

    def __len__(self):
        return len(self._image_names)


class ICCV09(Dataset):
    subsets = []
    default_dir = 'iccv09'

    def __init__(self, data_dir):  # TODO subset
        self._shape = [240, 320]
        self._images_dir = Path(f'{data_dir}/images')
        self._labels_dir = Path(f'{data_dir}/labels')
        self._image_list = [str(x)[:-4] for x in self._images_dir.iterdir()]

        info = {
            'problem':
                'semantic_segmentation',
            'class_count':
                8,
            'class_names': [
                'sky', 'tree', 'road', 'grass', 'water', 'building', 'mountain',
                'foreground object'
            ]
        }
        super().__init__(info=info)

    def get_example(self, idx):
        name = self._image_list[idx]

        def load_img():
            img = _load_image(self._images_dir / f"{name}.jpg")
            return tvtf.center_crop(img, self._shape)

        def load_lab():
            lab = np.loadtxt(
                self._labels_dir / f"{name}.regions.txt", dtype=np.int8)
            return numpy_transforms.center_crop(lab, self._shape, fill=-1)

        return _make_record(x_=load_img, y_=load_lab)

    def __len__(self):
        return len(self._image_list)


class VOC2012Segmentation(Dataset):
    subsets = ['train', 'val', 'trainval', 'test']
    default_dir = 'VOC2012'

    def __init__(self, data_dir, subset='train'):
        _check_subsets(self.__class__, subset)
        data_dir = Path(data_dir)

        sets_dir = data_dir / 'ImageSets/Segmentation'
        self._images_dir = data_dir / 'JPEGImages'
        self._labels_dir = data_dir / 'SegmentationClass'
        self._image_list = (sets_dir / f'{subset}.txt').read_text().splitlines()
        info = {
            'problem':
                'semantic_segmentation',
            'class_count':
                21,
            'class_names': [
                'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                'train', 'tvmonitor'
            ],
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
            img = _load_image(self._images_dir / f"{name}.jpg")
            return tvtf.center_crop(img, [500] * 2)

        def load_lab():
            lab = np.array(
                _load_image(self._labels_dir / f"{name}.png",
                            force_rgb=False).astype(np.int8))
            return numpy_transforms.center_crop(lab, [500] * 2, fill=-1)  # -1 ok?

        return _make_record(x_=load_img, y_=load_lab)

    def __len__(self):
        return len(self._image_list)


# Other


class ISUN(Dataset):
    # https://github.com/matthias-k/pysaliency/blob/master/pysaliency/external_datasets.py
    # TODO: labels, problem
    subsets = ['train', 'val', 'test']
    default_dir = 'iSUN'

    def __init__(self, data_dir, subset='train'):
        _check_subsets(self.__class__, subset)
        self._images_dir = f'{data_dir}/images'
        subset = {
            'train': 'training',
            'val': 'validation',
            'test': 'testing'
        }[subset]

        data_file = f'{data_dir}/{subset}.mat'
        data = loadmat(data_file)[subset]
        self._image_names = [d[0] for d in data['image'][:, 0]]

        super().__init__(subset=subset, info=dict(problem=None))

    def get_example(self, idx):
        return _make_record(
            x_=lambda: np.array(_load_image(f"{self._images_dir}/{self._image_names[idx]}.jpg")),
            y=-1)

    def __len__(self):
        return len(self._image_names)

"""
class LSUN(Dataset):
    # TODO: labels, replace with LSUNDatasetNew
    subsets = ['test']
    default_dir = 'LSUN'

    def __init__(self, data_dir, subset='train'):
        _check_subsets(self.__class__, subset)

        self._subset_dir = f'{data_dir}/{subset}'
        self._image_names = [
            os.path.relpath(x, start=self._subset_dir)
            for x in glob.glob(f'{self._subset_dir}/**/*.webp', recursive=True)
        ]
        super().__init__(subset=subset, info=dict(id='LSUN', problem=None))

    def get_example(self, idx):
        return _make_record(
            x_=lambda: np.array(_load_image(f"{self._subset_dir}/{self._image_names[idx]}")),
            y=-1)

    def __len__(self):
        return len(self._image_names)
"""