import torchvision.models as tvm
from pathlib import Path
from vidlu.utils.misc import download_if_not_downloaded
from vidlu.utils.path import read_lines
import re

name_to_url = {
    k: v for model_urls in [m.model_urls for m in vars(tvm).values() if hasattr(m, "model_urls")]
    for k, v in model_urls.items()}
name_to_url['deeplabv1_resnet101-coco'] = \
    "https://github.com/Ivan1248/semisup-seg-efficient/releases/download/pre-trained-dl/deeplabv1_resnet101-coco.pth"
#name_to_url['resnetv2-18'] = \
#    "https://github.com/Ivan1248/semisup-seg-efficient/releases/download/pre-trained-dl/resnetv2_18.pth"

name_to_file_name = dict()

_initialized = False


def add(url):
    from pathlib import Path
    path = Path(url)
    name_to_url[path.stem] = url


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

    sname = name[:name.rindex('.')] if re.match(r".*\.pth?$", name) else name
    if (url := name_to_url.get(sname, None)) is not None:
        path = pretrained_dir / Path(url).name
        download_if_not_downloaded(url, path)
    else:
        path = pretrained_dir / name_to_file_name.get(name, name)
    return path
