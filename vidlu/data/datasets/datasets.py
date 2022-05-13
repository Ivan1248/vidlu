import pickle
import json
import tarfile
import gzip
from pathlib import Path
import os
import shutil
import warnings
from vidlu.utils.func import partial

import PIL.Image as pimg
import numpy as np
from scipy.io import loadmat
import torch
import torchvision.datasets as dset
import torchvision.transforms.functional as tvtf
import torchvision.datasets.utils as tvdu
from vidlu.data import Dataset, Record
from vidlu.utils.misc import download, to_shared_array
from vidlu.transforms import numpy as numpy_transforms
from vidlu.utils.misc import extract_zip

from ._cityscapes_labels import labels as cslabels

# Constants

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


# Helper functions


def _load_image(path, force_rgb=True):
    img = pimg.open(path)
    if force_rgb and img.mode != 'RGB':
        img = img.convert('RGB')
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


def _check_size(*images, size, name=None):
    pre = "" if name is None else f"{name}: "
    if len(images) == 1:
        images = images[0]
        if not (len(images) == size):
            raise RuntimeError(
                f"{pre}The number of found images ({len(images)}) does not equal {size}.")
    else:
        images, labels = images
        if not (len(images) == len(labels) == size):
            raise RuntimeError(f"{pre}The number of found images ({len(images)}) or labels"
                               + f" ({len(labels)}) does not equal {size}.")


def load_image(path, downsampling):
    if not isinstance(downsampling, int):
        raise ValueError("`downsampling` must be an `int`.")
    img = _load_image(path)
    if downsampling > 1:
        img = tvtf.resize(img, tuple(np.flip(img.size) // downsampling), pimg.BILINEAR)
    return img


def load_segmentation(path, downsampling, id_to_label=None, dtype=np.int8):
    """Loads and optionally translates segmentation labels.

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
        lab = tvtf.resize(lab, tuple(np.flip(lab.size) // downsampling), pimg.NEAREST)

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

    def __init__(self, distribution='normal', mean=0, std=1, example_shape=(32, 32, 3), size=50000,
                 seed=53, key='image'):
        self._shape = example_shape
        self._rand = np.random.RandomState(seed=seed)
        self._seeds = self._rand.randint(1, size=(size,))
        if distribution not in ('normal', 'uniform'):
            raise ValueError('Distribution not in {"normal", "uniform"}')
        self._distribution = distribution
        self.mean = mean
        self.std = std
        self.key = key
        super().__init__(name=f'WhiteNoise-{distribution}({mean},{std},{example_shape})',
                         subset=f'{seed}{size}', data=self._seeds)

    def get_example(self, idx):
        self._rand.seed(self._seeds[idx])
        if self._distribution == 'normal':
            return _make_record(**{self.key: self._rand.randn(*self._shape) * self.std + self.mean})
        elif self._distribution == 'uniform':
            d = 12 ** 0.5 / 2
            raise NotImplementedError("mean, std")
            return _make_record(**{self.key: self._rand.uniform(-d, d, self._shape)})


class RademacherNoise(Dataset):
    subsets = []

    def __init__(self, shape=(32, 32, 3), size=50000, seed=53, key='image'):
        # lambda: np.random.binomial(n=1, p=0.5, size=(ood_num_examples, 3, 32, 32)) * 2 - 1
        self._shape = shape
        self._rand = np.random.RandomState(seed=seed)
        self._seeds = self._rand.randint(1, size=(size,))
        self.key = key
        super().__init__(
            name=f'RademacherNoise{shape}',
            subset=f'{seed}-{size}',
            data=self._seeds)

    def get_example(self, idx):
        self._rand.seed(self._seeds[idx])
        return _make_record(**{self.key: self._rand.binomial(n=1, p=0.5, size=self._shape)})


class HBlobs(Dataset):
    subsets = []

    def __init__(self, sigma=None, shape=(32, 32, 3), size=50000, seed=53, key='image'):
        # lambda: np.random.binomial(n=1, p=0.5, size=(ood_num_examples, 3, 32, 32)) * 2 - 1
        self._shape = shape
        self._rand = np.random.RandomState(seed=seed)
        self._seeds = self._rand.randint(2 ** 31, size=(size,))
        self._sigma = sigma or 1.5 * shape[0] / 32
        self.key = key
        super().__init__(name=f'HBlobs({shape})', subset=f'{seed}-{size}', data=self._seeds)

    def get_example(self, idx):
        from skimage.filters import gaussian
        self._rand.seed(self._seeds[idx])
        x = self._rand.binomial(n=1, p=0.7, size=self._shape)
        x = gaussian(np.float32(x), sigma=self._sigma, multichannel=False)
        x[x < 0.75] = 0
        return _make_record(**{self.key: x})


class Blank(Dataset):
    subsets = []

    def __init__(self, value=0, shape=(32, 32, 3), size=50000, key='image'):
        self._shape = shape
        self._len = size
        self.value = value
        self.key = key
        super().__init__(name=f'Blank({shape},{value})', subset=f'{size}')

    def get_example(self, idx):
        return _make_record(**{self.key: np.full(self._shape, self.value)})

    def __len__(self):
        return self._len


class DummyClassification(Dataset):
    subsets = []

    def __init__(self, shape=(28, 28, 3), size=256, key='image'):
        self._shape = shape
        self._colors = [row for row in np.eye(3, 3) * 255]
        self._len = size
        super().__init__(name=f'DummyClassification({shape})', subset=f'{size}',
                         info=dict(class_count=len(self._colors), problem='classification'))

    def get_example(self, idx):
        color_idx = idx % len(self._colors)
        return _make_record(**{self.key: np.ones(self._shape) * self._colors[color_idx]},
                            class_label=color_idx)

    def __len__(self):
        return self._len


# ImageFolder

class ImageFolder(Dataset):
    def __init__(self, data_dir, subset='all'):
        self.data_dir = Path(data_dir)
        subset_dir = self.data_dir if subset == 'all' else self.data_dir / subset
        self._elements = sorted(p.name for p in subset_dir.iterdir())
        super().__init__(name=f"imageFolder{self.data_dir.name}", subset=subset,
                         info=dict(problem='images'))

    def get_example(self, idx):
        return _make_record(image_=lambda: _load_image(self.data_dir / self._elements[idx]))


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
        x, y = self.load_array(x_path, is_image=True), self.load_array(y_path, is_image=False)
        self.x, self.y = map(to_shared_array, [x, y])
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
    def load_array(path, is_image):
        with open(path, 'rb') as f:
            return (np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28) if is_image
                    else np.frombuffer(f.read(), np.uint8, offset=8))

    def get_example(self, idx):
        return _make_record(image=self.x[idx], class_label=self.y[idx])

    def __len__(self):
        return len(self.y)


class SVHN(Dataset):
    subsets = ['trainval', 'test']
    default_dir = 'SVHN'

    def __init__(self, data_dir, subset='trainval'):
        _check_subsets(self.__class__, subset)
        ss = 'train' if subset == 'trainval' else subset
        data = loadmat(Path(data_dir) / (ss + '_32x32.mat'))
        self.x, self.y = data['X'], np.remainder(data['y'], 10)
        super().__init__(subset=subset, info=dict(class_count=10, problem='classification'))

    def get_example(self, idx):
        return _make_record(image=self.x[idx], class_label=self.y[idx])

    def __len__(self):
        return len(self.x)


def unpickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f, encoding='latin1')


class Cifar10(Dataset):
    subsets = ['trainval', 'test']  # TODO: use original subset names
    default_dir = 'cifar-10-batches-py'
    info = dict(
        class_count=10, problem='classification', in_ram=True,
        class_names=["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship",
                     "truck"])

    def __init__(self, data_dir, subset='trainval'):
        _check_subsets(self.__class__, subset)
        data_dir = Path(data_dir)

        self.download_if_necessary(data_dir)

        h, w, ch = 32, 32, 3
        if subset == 'trainval':
            x, y = [], []
            for i in range(1, 6):
                data = unpickle(data_dir / f'data_batch_{i}')
                x.append(data['data'])
                y += data['labels']
            x = np.vstack(x).reshape((-1, ch, h, w)).transpose(0, 2, 3, 1)
            y = np.array(y, dtype=np.int8)
        else:  # ss == 'test':
            data = unpickle(data_dir / 'test_batch')
            x = data['data'].reshape((-1, ch, h, w)).transpose(0, 2, 3, 1)
            y = np.array(data['labels'], dtype=np.int8)
        self.x, self.y = map(to_shared_array, [x, y])
        super().__init__(subset=subset, info=Cifar10.info)

    def download(self, data_dir):
        datasets_dir = data_dir.parent
        download_path = datasets_dir / "cifar-10-python.tar.gz"
        print(f"Downloading dataset to {datasets_dir}")
        download(url="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
                 output_path=download_path, md5='c58f30108f718f92721af3b95e74349a')
        with tarfile.open(download_path, "r:gz") as tar:
            tar.extractall(path=datasets_dir)
        download_path.unlink()

    def get_example(self, idx):
        return _make_record(image=self.x[idx], class_label=self.y[idx])

    def __len__(self):
        return len(self.x)


class Cifar100(Dataset):
    subsets = ['trainval', 'test']
    default_dir = 'cifar-100-python'

    def __init__(self, data_dir, subset='trainval'):
        _check_subsets(self.__class__, subset)
        data_dir = Path(data_dir)

        self.download_if_necessary(data_dir)

        data = unpickle(data_dir / f"{'train' if subset == 'trainval' else subset}")

        h, w, ch = 32, 32, 3
        train_x = data['data'].reshape((-1, ch, h, w)).transpose(0, 2, 3, 1)
        self.x, self.y = train_x, data['fine_labels']

        super().__init__(subset=subset, info=dict(class_count=100, problem='classification',
                                                  coarse_labels=data['coarse_labels']))

    def download(self, data_dir):
        datasets_dir = data_dir.parent
        download_path = datasets_dir / "cifar-100-python.tar.gz"
        print(f"Downloading dataset to {datasets_dir}")
        download(url="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
                 output_path=download_path, md5='eb9058c3a382ffc7106e4002c42a8d85')
        with tarfile.open(download_path, "r:gz") as tar:
            tar.extractall(path=datasets_dir)
        download_path.unlink()

    def get_example(self, idx):
        return _make_record(image=self.x[idx], class_label=self.y[idx])

    def __len__(self):
        return len(self.x)


class DescribableTextures(Dataset):
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
        return _make_record(image=x, class_label=y)


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
        return _make_record(image_=lambda: _load_image(img_path), class_label=lab)

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
        img_path = self._data_dir / self._file_names[idx]
        return _make_record(image_=load_image(img_path, self._downsampling),
                            class_label=self._labels[idx])

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
        return _make_record(image_=lambda: self.load_image(idx), class_label=-1)

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

    def __init__(self, data_dir, load_func, extensions=None, is_valid_file=None, input_key='image',
                 **kwargs):
        self.root = data_dir
        classes, class_name_to_idx = self._find_classes(data_dir)
        self.path_to_class = _read_classification_dataset(data_dir, class_name_to_idx, extensions,
                                                          is_valid_file)
        if len(self.path_to_class) == 0:
            raise RuntimeError("Found 0 files in subfolders of: " + self.root
                               + "\nSupported extensions are: " + ",".join(extensions) + ".")
        self.load = load_func
        self.input_key = input_key
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
        return _make_record(**{self.input_key: x}, class_label=y)

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
            image_=lambda: np.array(
                _load_image(f"{self._images_dir}/{self._image_names[idx]}.jpg")),
            class_label=-1)

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
            image_=lambda: np.array(_load_image(f"{self._subset_dir}/{self._image_names[idx]}")),
            class_label=-1)

    def __len__(self):
        return len(self._image_names)
"""


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
        'Void': {'Void': (0, 0, 0)}}
    color_to_label = {color: i for i, class_name_color in enumerate(class_groups_colors.values())
                      for _, color in class_name_color.items()}
    color_to_label[(0, 0, 0)] = -1
    info = dict(
        problem='semantic_segmentation', class_count=11,
        class_names=list(class_groups_colors.keys()),
        class_colors=[next(iter(v.values())) for v in class_groups_colors.values()],
        url="https://github.com/Ivan1248/CamVid/archive/master.zip")

    def download(self, data_dir):
        datasets_dir = Path(data_dir).parent
        download_path = datasets_dir / "CamVid.zip"
        download(url=self.info["url"], output_path=download_path)
        print(f"Extracting dataset to {datasets_dir}")
        extract_zip(download_path, datasets_dir)
        shutil.move(datasets_dir / 'CamVid-master', data_dir)
        download_path.unlink()

    def __init__(self, data_dir, subset='train', downsampling=1):
        _check_subsets(self.__class__, subset)
        if downsampling < 1:
            raise ValueError("downsampling must be greater or equal to 1.")

        data_dir = Path(data_dir)
        self.download_if_necessary(data_dir)

        self._downsampling = downsampling

        img_dir, lab_dir = data_dir / '701_StillsRaw_full', data_dir / 'LabeledApproved_full'
        self._img_lab_list = [(str(img_dir / f'{name}.png'), str(lab_dir / f'{name}_L.png'))
                              for name in (data_dir / f'{subset}.txt').read_text().splitlines()]
        super().__init__(
            subset=subset if downsampling == 1 else f"{subset}.downsample({downsampling})",
            info=self.info)

    def get_example(self, idx):
        ip, lp = self._img_lab_list[idx]
        ds = self._downsampling
        return _make_record(
            image_=lambda: load_image(ip, ds),
            seg_map_=lambda: load_segmentation(lp, ds, self.color_to_label),
            name=ip)

    def __len__(self):
        return len(self._img_lab_list)


class CamVidSequences(Dataset):  # TODO
    subset_to_size = {'06R0': 3001, '16E5': 8251, '01TP': 3691, 'Seq05VD': 5101}
    subsets = list(subset_to_size.keys())  # 01TP and Seq05VD contain the test set
    default_dir = 'CamVid-sequences'

    def download(self, data_dir):
        raise NotImplementedError()

    def __init__(self, data_dir, subset, downsampling=1):
        _check_subsets(self.__class__, subset)
        if downsampling < 1:
            raise ValueError("downsampling must be greater or equal to 1.")

        data_dir = Path(data_dir)
        self.download_if_necessary(data_dir)

        self._downsampling = downsampling

        self._image_paths = list((data_dir / subset).iterdir())
        _check_size(self._image_paths, size=self.subset_to_size[subset],
                    name=f"{type(self).__name__}-{subset}")

        info = dict(
            problem='semantic_segmentation', class_count=11,
            class_names=list(CamVid.class_groups_colors.keys()),
            class_colors=[next(iter(v.values())) for v in CamVid.class_groups_colors.values()])

        super().__init__(
            subset=subset if downsampling == 1 else f"{subset}.downsample({downsampling})",
            info=info)

    def get_example(self, idx):
        image_path = self._image_paths[idx]
        return _make_record(image_=lambda: load_image(image_path, self._downsampling),
                            name=image_path)

    def __len__(self):
        return len(self._image_paths)


_cityscapes_class_names = [
    'road',  # 0
    'sidewalk',  # 1
    'building',  # 2
    'wall',  # 3
    'fence',  # 4
    'pole',  # 5
    'traffic light',  # 6
    'traffic sign',  # 7
    'vegetation',  # 7
    'terrain',  # 9
    'sky',  # 10
    'person',  # 11
    'rider',  # 12
    'car',  # 13
    'truck',  # 14
    'bus',  # 15
    'train',  # 16
    'motorcycle',  # 17
    'bicycle'  # 18
]


class Cityscapes(Dataset):
    subsets = ['train', 'val', 'test', 'train_extra']  # 'test' labels are invalid
    default_dir = 'Cityscapes'
    subset_to_size = dict(train=2975, val=500, test=1525, train_extra=19998)
    info = dict(problem='semantic_segmentation', class_count=19,
                class_names=[l.name for l in cslabels if l.trainId >= 0],
                class_colors=[l.color for l in cslabels if l.trainId >= 0],
                in_ram=False)  # vup.get_partition(data_dir) == 'tmpfs'

    def __init__(self, data_dir, subset='train', label_kind=None, downsampling=1,
                 downsample_labels=True):
        if label_kind is None:
            label_kind = 'gtCoarse' if subset == 'train_extra' else 'gtFine'
        data_dir = Path(data_dir)
        im_data_dir = data_dir / subset if subset == 'train_extra' else data_dir
        _check_subsets(self.__class__, subset)
        if downsampling < 1:
            raise ValueError("downsampling must be greater or equal to 1.")

        self.downsampling, self.downsample_labels = downsampling, downsample_labels

        subdirs = {x.name for x in im_data_dir.glob('*')}
        if 'leftImg8bit' in subdirs:
            images_dir, labels_dir = 'leftImg8bit', label_kind
            img_suffix, lab_suffix = "_leftImg8bit.png", f"_{label_kind}_labelIds.png"
            self._id_to_label = {l.id: l.trainId for l in cslabels}
        elif 'rgb' in subdirs:  # IK's format
            images_dir, labels_dir, img_suffix, lab_suffix = 'rgb', 'labels', ".ppm", ".png"
            self._id_to_label = {**{i: i for i in range(19)}, 19: -1}
        else:
            raise RuntimeError(f"Invalid directory structure in {im_data_dir}: {subdirs=}.")

        self._images_dir = im_data_dir / images_dir / subset
        self._labels_dir = data_dir / labels_dir / subset

        self._images = list(sorted([
            x.relative_to(self._images_dir) for x in self._images_dir.glob('*/*')]))
        self._labels = [str(x)[:-len(img_suffix)] + lab_suffix for x in self._images]

        _check_size(self._images, size=self.subset_to_size[subset],
                    name=f"{type(self).__name__}-{subset}")

        super().__init__(
            subset=subset if downsampling == 1 else f"{subset}.downsample({downsampling})",
            info=self.info)

    def get_example(self, idx):
        im_path = self._images_dir / self._images[idx]
        lab_path = self._labels_dir / self._labels[idx]
        d = self.downsampling
        im_get = lambda: load_image(im_path, d)
        lab_get = lambda: load_segmentation(
            lab_path, d if self.downsample_labels else 1, self._id_to_label)
        return _make_record(image_=im_get,
                            seg_map_=lab_get,
                            name=str(self._images[idx].with_suffix("")))

    def __len__(self):
        return len(self._images)


class WildDash(Dataset):
    subsets = ['val', 'bench']
    splits = dict(all=(('val', 'bench'), None), both=(('val', 'bench'), None))
    default_dir = 'WildDash'

    def __init__(self, data_dir, subset='val', downsampling=1):
        _check_subsets(self.__class__, subset)
        if downsampling < 1:
            raise ValueError("downsampling must be greater or equal to 1.")

        self._subset = subset

        self._downsampling = downsampling
        self._shape = np.array([1070, 1920]) // downsampling

        self._IMG_SUFFIX = "0.png"
        self._LAB_SUFFIX = "0_labelIds.png"
        self._id_to_label = [(l.id, l.trainId) for l in cslabels]

        self._images_dir = Path(f'{data_dir}/wd_{subset}_01')
        self._image_names = sorted([
            str(x.relative_to(self._images_dir))[:-5]
            for x in self._images_dir.glob(f'*{self._IMG_SUFFIX}')
        ])

        info = dict(problem='semantic_segmentation',
                    class_count=19,
                    class_names=[l.name for l in cslabels if l.trainId >= 0],
                    class_colors=[l.color for l in cslabels if l.trainId >= 0])
        self._blank_label = np.full(list(self._shape), -1, dtype=np.int8)
        super().__init__(
            subset=subset if downsampling == 1 else f"{subset}.downsample({downsampling})",
            info=info)

    def get_example(self, idx):
        path_prefix = f"{self._images_dir}/{self._image_names[idx]}"

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

        return _make_record(
            image_=partial(load_image, f"{path_prefix}{self._IMG_SUFFIX}", self._downsampling),
            seg_map_=load_lab)

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

        info = dict(problem='semantic_segmentation', class_count=8,
                    class_names=['sky', 'tree', 'road', 'grass', 'water', 'building', 'mountain',
                                 'foreground object'])
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

        return _make_record(image_=load_img, seg_map_=load_lab)

    def __len__(self):
        return len(self._image_list)


class VOC2012Segmentation(Dataset):
    subset_to_size = {'train': 1464, 'val': 1449, 'test': None, 'trainval': 2913,
                      'train_aug': 10582}
    subsets = tuple(subset_to_size.keys())
    default_dir = 'VOCdevkit'
    subdir = 'VOC2012'
    info = dict(
        problem='semantic_segmentation',
        class_count=21,
        class_names=['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                     'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                     'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
        class_colors=[(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
                      (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
                      (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0),
                      (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (0, 0, 230),
                      (0, 0, 230), (119, 11, 32)],
        url=r'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        aug_urls=dict(  # train_aug is a superset of train and does not overlap with val
            labels=r'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/SegmentationClassAug.zip',
            train_list=r'http://raw.githubusercontent.com/hfslyc/AdvSemiSeg/master/dataset/voc_list/train_aug.txt',
        ),
        md5='6cd6e144f989b92b3379bac3b3de84fd')

    def __init__(self, data_dir, subset='train', pad=False):
        _check_subsets(self.__class__, subset)
        data_dir = Path(data_dir)
        self.pad = pad

        self.download_if_necessary(data_dir)

        data_subdir = data_dir / self.subdir
        self._images_dir = data_subdir / 'JPEGImages'
        if subset == 'train_aug':
            sets_dir = data_subdir / 'ImageSets/SegmentationAug'
            self._labels_dir = data_subdir / 'SegmentationClassAug'
            if not sets_dir.exists():
                self.download_aug(data_subdir)
        else:
            sets_dir = data_subdir / 'ImageSets/Segmentation'
            self._labels_dir = data_subdir / 'SegmentationClass'
        self._image_list = (sets_dir / f'{subset}.txt').read_text().splitlines()

        super().__init__(subset=subset, info=self.info)

    def download_aug(self, root_dir):
        tvdu.download_and_extract_archive(self.info['aug_urls']['labels'], root_dir,
                                          remove_finished=True)
        lists_dir = root_dir / 'ImageSets' / 'SegmentationAug'
        os.makedirs(lists_dir, exist_ok=True)
        if not (file_path := lists_dir / f'train_aug.txt').exists():
            download(self.info['aug_urls'][f'train_list'], file_path)

    def get_example(self, idx):
        name = self._image_list[idx]

        def load_img():
            img = _load_image(self._images_dir / f"{name}.jpg")
            return tvtf.center_crop(img, [500] * 2) if self.pad else img

        def load_lab():
            lab = np.array(_load_image(self._labels_dir / f"{name}.png", force_rgb=False)) \
                .astype(np.int8)
            return numpy_transforms.center_crop(lab, [500] * 2,
                                                fill=-1) if self.pad else lab  # -1 ok?

        return _make_record(image_=load_img, seg_map_=load_lab)

    def __len__(self):
        return len(self._image_list)

# class Viper(Dataset):
#     class_info = _viper_mapping.get_class_info()
#
#     mapping = np.full(len(_viper_mapping.labels), ignore_id)
#     for i, city_id in enumerate(_viper_mapping.get_train_ids()):
#         mapping[city_id] = i
#
#     def __init__(self, args, subset='train', train=False):
#         super().__init__(args, train)
#         data_dir = Path(args.data_path) / 'viper' / subset / 'img'
#
#         self.img_paths = []
#         for city in next(os.walk(data_dir))[1]:
#             city_path = data_dir / city
#             files = next(os.walk(city_path))[2]
#             self.img_paths.extend(city_path / f for f in files)
#
#         self.names = [f.with_suffix('') for f in files]
#
#         self.label_paths = {f: f.replace('img', 'cls').replace('jpg', 'png') for f in self.img_paths
#                             if os.path.exists(f.replace('img', 'cls').replace('jpg', 'png'))}
#         print('\nTotal num images =', len(self.img_paths))
#
#     def __len__(self):
#         return len(self.img_paths)
#
#     def get_example(self, idx):
#         img_path = self.img_paths[idx]
#         img = pimg.open(img_path)
#
#         batch = {}
#
#         if img_path in self.label_paths.keys():
#             labels = pimg.open(self.label_paths[img_path])
#
#         img_width, img_height = img.size
#
#         if self.reshape_size > 0:
#             smaller_side = min(img_width, img_height)
#             scale = float(self.reshape_size) / smaller_side
#         else:
#             scale = 1
#
#         img_size = (int(img_width * scale), int(img_height * scale))
#         if self.train:
#             scale = np.random.uniform(
#                 self.min_jitter_scale, self.max_jitter_scale)
#             img_size = (round(scale * img_size[0]), round(scale * img_size[1]))
#
#         img_size = transform.pad_size_for_pooling(
#             img_size, self.last_block_pooling)
#         img = transform.resize_img(img, img_size)
#         if labels is not None:
#             labels = transform.resize_labels(labels, img_size)
#
#         img = np.array(img, dtype=np.float32)
#         if labels is not None:
#             labels = np.array(labels, dtype=np.int64)
#             labels = self.mapping[labels]
#
#         if self.train:
#             img = transform.pad(img, self.crop_size, 0)
#             labels = transform.pad(labels, self.crop_size, self.ignore_id)
#
#         batch['mean'] = self.mean
#         batch['std'] = self.std
#         img = transform.normalize(img, self.mean, self.std)
#
#         img = transform.numpy_to_torch_image(img)
#         batch['image'] = img
#
#         batch['name'] = self.names[idx]
#         if labels is not None:
#             labels = torch.LongTensor(labels)
#             batch['labels'] = labels
#             batch['target_size'] = labels.shape[:2]
#         else:
#             batch['target_size'] = img.shape[:2]
#
#         return batch
#
# Other

# class DarkZurich(Dataset):
#     subsets = ['train', 'val', 'val_ref']
#     default_dir = 'dark_zurich'
#
#     def __init__(self, data_dir, subset='train', downsampling=1):
#         _check_subsets(self.__class__, subset)
#         if downsampling < 1:
#             raise ValueError("downsampling must be greater or equal to 1.")
#
#         data_dir = Path(data_dir)
#
#         self._downsampling = downsampling
#
#         corresp_dir = data_dir / "corresp" / subset
#         gps_dir = data_dir / "gps" / subset
#         rgb_anon_dir = data_dir / "gps" / subset
#
#         img_dir = data_dir / '701_StillsRaw_full'
#         lab_dir = data_dir / 'LabeledApproved_full'
#         lines = (data_dir / f'{subset}.txt').read_text().splitlines()
#         self._img_lab_list = [(str(img_dir / f'{x}.png'), str(lab_dir / f'{x}_L.png'))
#                               for x in lines]
#         # info = dict(
#         #     problem='semantic_segmentation',
#         #     class_count=11,
#         #     class_names=list(CamVid.class_groups_colors.keys()),
#         #     class_colors=[next(iter(v.values())) for v in
#         #                   CamVid.class_groups_colors.values()])
#         #
#         # self.color_to_label = dict()
#         # for i, class_name_color in enumerate(CamVid.class_groups_colors.values()):
#         #     for _, color in class_name_color.items():
#         #         self.color_to_label[color] = i
#         # self.color_to_label[(0, 0, 0)] = -1
#         #
#         # modifiers = [f"downsample({downsampling})"] if downsampling > 1 else []
#         # super().__init__(subset=subset, modifiers=modifiers, info=info)
#
#     def get_example(self, idx):
#         ip, lp = self._img_lab_list[idx]
#         # df = self._downsampling
#         # return _make_record(
#         #     image_=lambda: load_image_with_downsampling(ip, df),
#         #     seg_map_=lambda: load_segmentation_with_downsampling(lp, df, self.color_to_label))
#
#     def __len__(self):
#         return len(self._img_lab_list)
