from pathlib import Path
import re
import torchvision.models as tvm

from vidlu.utils.misc import download_if_not_downloaded, deep_getattr
from vidlu.utils.path import read_lines

_legacy_name_to_url = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'deeplabv1_resnet101-coco':
        'https://github.com/Ivan1248/semisup-seg-efficient/releases/download/pre-trained-dl/deeplabv1_resnet101-coco.pth',
    'hardnet68':
        'https://github.com/PingoLH/Pytorch-HarDNet/blob/master/hardnet68.pth?raw=True',
    # hardnet-petite? https://github.com/Ivan1248/FCHarDNet
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
}
# name_to_url['resnetv2-18'] = \
#    "https://github.com/Ivan1248/semisup-seg-efficient/releases/download/pre-trained-dl/resnetv2_18.pth"

name_to_file_name = dict()

_initialized = False


def add(url):
    from pathlib import Path
    path = Path(url)
    _legacy_name_to_url[path.stem] = url


def initialize(pretrained_dir):
    pretrained_dir = Path(pretrained_dir)
    if (urls_file_path := pretrained_dir / "urls.txt").exists():
        for line in read_lines(urls_file_path):
            add(line)
    for fpath in pretrained_dir.glob("**/*"):
        if fpath.is_file():
            name_to_file_name[fpath.stem] = fpath.name


def get_path(name: str, pretrained_dir) -> Path:
    pretrained_dir = Path(pretrained_dir)
    if not _initialized:
        initialize(pretrained_dir)
    
    if re.match(r"https?://.*", name):
        url = name
    else:
        sname = name[:name.rindex('.')] if re.match(r".*\.pth?$", name) else name
        url = _legacy_name_to_url.get(sname, None)
        if url is None:
            import torchvision.models
            try:
                url = deep_getattr(torchvision.models, sname).url
            except AttributeError:
                pass
    if url is not None:
        fname = Path(url).name
        if "?" in fname:
            fname = fname[:fname.index('?')]
        path = pretrained_dir / fname
        download_if_not_downloaded(url, path)
    else:
        path = pretrained_dir / name_to_file_name.get(name, name)
    return path
