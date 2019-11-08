import pickle
import json
import tarfile
import gzip
import zipfile
from pathlib import Path
import os
import shutil
import warnings

import PIL.Image as pimg
import numpy as np
from scipy.io import loadmat
import torch
import torchvision.datasets as dset
import torchvision.transforms.functional as tvtf
from tqdm import tqdm

from .. import Dataset, Record
from vidlu.utils.misc import download, to_shared_array
from vidlu.transforms import numpy as numpy_transforms

from ._cityscapes_labels import labels as cslabels

# Constants

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


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
        if (k == 'x' and isinstance(v, np.ndarray) and v.dtype == np.uint8
                and 2 <= len(v.shape) <= 3):
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


def load_segmentation_with_downsampling(path, downsampling, id_to_label=None,
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
    if id_to_label is None:
        id_to_label = dict()
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
        for id_, lb in id_to_label.items():
            lab[lab == id_] = lb
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

    def __init__(self, example_shape=(28, 28, 3), size=256):
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


# ImageFolder

class ImageFolder(Dataset):
    def __init__(self, data_dir, subset='all'):
        self.data_dir = Path(data_dir)
        subset_dir = self.data_dir if subset == 'all' else self.data_dir / subset
        self._elements = sorted(p.name for p in subset_dir.iterdir())
        super().__init__(subset=subset, info=dict(problem='images'), modifiers=self.data_dir.name)

    def get_example(self, idx):
        return _make_record(x_=lambda: _load_image(self.data_dir / self._elements[idx]))


# Images

class CamVidSequences(ImageFolder):
    subsets = ['01TP', '01TP', '05VD', '06RO', '16E4']


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
        self.x, self.y = map(to_shared_array, [self.x, self.y])
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

    def __init__(self, data_dir, subset='trainval', random_labels=False):
        self.random_labels = random_labels
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
        self.x, self.y = map(to_shared_array, [self.x, self.y])
        super().__init__(subset=subset,
                         info=dict(class_count=10, problem='classification', in_ram=True),
                         modifiers=['random_labels'] if self.random_labels else None)

    def get_example(self, idx):
        if self.random_labels:
            return _make_record(x=self.x[idx], y=self.y[(idx + 1) % len(self)])
        else:
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
        super().__init__(subset=ss, info=dict(class_count=47),
                         data=dset.ImageFolder(f"{data_dir}/images"))
        self.data = dset.ImageFolder(f"{data_dir}/images")
        raise NotImplementedError("DescribableTexturesDataset not implemented")

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
        # TODO: use superspecies parameter
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
                return np.fromstring(data, dtype='uint8').reshape((32, 32, 3), order="F")

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


def _read_classification_dataset(data_dir, class_name_to_idx, extensions=None, is_valid_file=None):
    path_to_class = []
    data_dir = Path(data_dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError(
            "Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        is_valid_file = lambda x: x.lower().endswith(extensions)
    for target in sorted(class_name_to_idx.keys()):
        d = data_dir / target
        if not d.is_dir():
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = Path(root, fname)
                if is_valid_file(path):
                    item = (path, class_name_to_idx[target])
                    path_to_class.append(item)

    return path_to_class


class ClassificationFolderDataset(Dataset):  # TODO
    """A generic data loader where the examples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Copied from Torchvision and modified.

    Args:
        root (string): Root directory path.
        load_func (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        path_to_class (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, data_dir, load_func, extensions=None, is_valid_file=None, **kwargs):
        classes, class_name_to_idx = self._find_classes(data_dir)
        self.path_to_class = _read_classification_dataset(data_dir, class_name_to_idx, extensions,
                                                          is_valid_file)
        if len(self.path_to_class) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root
                                + "\nSupported extensions are: " + ",".join(extensions)) + ".")
        self.load = load_func
        super().__init__(**kwargs)
        self.info.classes, self.info.class_name_to_idx = classes, class_name_to_idx

    @staticmethod
    def _find_classes(data_dir):
        """Finds the class folders in a dataset.

        Args:
            data_dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir),
            and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(data_dir) if d.is_dir()]
        classes.sort()
        class_name_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_name_to_idx

    def get_example(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            Record: (sample, target) where target is class_index of the target class.
        """
        path, y = self.path_to_class[idx]
        x = self.load(path)
        return _make_record(x_=x, y=y)

    def __len__(self):
        return len(self.path_to_class)


class ImageClassificationFolderDataset(ClassificationFolderDataset):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        load_func (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, load_func=_load_image, is_valid_file=None, **kwargs):
        super().__init__(root, load_func,
                         IMG_EXTENSIONS if is_valid_file is None else None,
                         is_valid_file=is_valid_file)


class ImageNet(ImageClassificationFolderDataset):  # TODO
    """`ImageNet <http://image-net.org/>`_ 2012 Classification Dataset.

    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """
    subsets = ["train", "val"]
    default_dir = 'tiny-images'

    def __init__(self, root, subset='train', **kwargs):
        # TODO: use subset parameter
        root = self.root = os.path.expanduser(root)

        wnid_to_classes = torch.load(self.meta_file)[0]

        super().__init__(self.subset_folder, **kwargs)
        self.root = root

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {
            cls: idx for idx, clss in enumerate(self.classes) for cls in clss}

    @property
    def meta_file(self):
        return os.path.join(self.root, 'meta.bin')

    @property
    def subset_folder(self):
        return os.path.join(self.root, self.subset)


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
        for i, class_name_color in enumerate(CamVid.class_groups_colors.values()):
            for _, color in class_name_color.items():
                self.color_to_label[color] = i
        self.color_to_label[(0, 0, 0)] = -1

        modifiers = [f"downsample({downsampling})"] if downsampling > 1 else []
        super().__init__(subset=subset, modifiers=modifiers, info=info)

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

        img_suffix = "_leftImg8bit.png"
        lab_suffix = "_gtFine_labelIds.png"
        self._id_to_label = {l.id: l.trainId for l in cslabels}

        self._images_dir = Path(f'{data_dir}/leftImg8bit/{subset}')
        self._labels_dir = Path(f'{data_dir}/gtFine/{subset}')
        self._image_list = [x.relative_to(self._images_dir) for x in self._images_dir.glob('*/*')]
        self._image_list = list(sorted(self._image_list))
        self._label_list = [str(x)[:-len(img_suffix)] + lab_suffix for x in self._image_list]

        info = dict(problem='semantic_segmentation', class_count=19,
                    class_names=[l.name for l in cslabels if l.trainId >= 0],
                    class_colors=[l.color for l in cslabels if l.trainId >= 0])
        modifiers = [f"downsample({downsampling})"] if downsampling > 1 else []
        super().__init__(subset=subset, modifiers=modifiers, info=info)

    def get_example(self, idx):
        im_path = self._images_dir / self._image_list[idx]
        lab_path = self._labels_dir / self._label_list[idx]
        d = self._downsampling
        return _make_record(
            x_=lambda: load_image_with_downsampling(im_path, d),
            y_=lambda: load_segmentation_with_downsampling(lab_path, d, self._id_to_label))

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
        modifiers = [f"downsample({downsampling})"] if downsampling > 1 else []
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
            for id_, lb in self._id_to_label:
                lab[lab == id_] = lb
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
