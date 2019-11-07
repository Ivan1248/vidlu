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
    path = str(path).strip()
    allowed = r"-\w.,\'\\/!#$%^&()_+=@{}\[\]"
    return Path(re.sub(f"(?u)[^{allowed}]", "+", str(path)))


def get_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def dir_tree_has_no_files(path: Path):
    return path.is_dir() and all(dir_tree_has_no_files(c) for c in path.iterdir())


def remove_dir_trees_with_no_files(path: Path, verbose=False):
    children = list(path.iterdir())
    if path.is_dir() and all(remove_dir_trees_with_no_files(c, verbose) for c in children):
        if verbose:
            print('Deleting', path)
        path.rmdir()
        return True
    return False


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


def read_text(path):
    return Path(path).open("r").read()


def read_lines(path):
    return Path(path).open("r").readlines()


def write_text(path, text, append=False):
    Path(path).open("w+" if append else "w").write(text)


def write_lines(path, lines, append=False):
    with open(path, "w+" if append else "w") as file:
        file.writelines(lines)
