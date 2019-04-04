import os
import tempfile
from pathlib import Path
import re


# Path #############################################################################################

def find_in_ancestor(path, ancestor_sibling_path):
    """
    `ancestor_sibling_path` can be the name of a sibling directory (or some
    descendant of the sibling) to some ancestor of `path` or `path`
    """
    path = Path(path).absolute()
    for anc in path.parents:
        candidate = anc / ancestor_sibling_path
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No ancestor sibling found")


def to_valid_path(path):
    path = str(path).strip().replace('"', "'")
    return Path(re.sub(r'(?u)[^-\w.{}[\]+=\'\\/]', "+", path))


def get_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


# File #############################################################################################


def create_file_atomic(path, save_action):
    """
    Performs `save_action` on a temporary file and, if successful, moves the
    temporary file to `path`. This avoids creation of incomplete files if an
    error during writing occurs.
    Args:
        path: The path of the output file.
        save_action: A procedure that accepts a file as the only argument.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, dir=Path(path).parent)
    try:
        save_action(tmp.file)
    except BaseException:
        tmp.close()
        os.remove(tmp.name)
        raise
    else:
        tmp.close()
        os.rename(tmp.name, path)


def write_text(path, text, append=False):
    Path(path).open("w").write(text)


def write_lines(path, lines, append=False):
    with open(path, "w+" if append else "w") as file:
        file.writelines(lines)
