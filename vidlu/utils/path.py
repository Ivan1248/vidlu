import functools
import os
import tempfile
from pathlib import Path
import re
import datetime as dt


# Path #############################################################################################

def find_in_directories(dirs, child):
    for candidate in (Path(d) / child for d in dirs):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"None of {dirs} contains '{child}'.")


def find_in_ancestors(start, subpath, include_start=False, ignore_broken_symlinks=False):
    """
    `ancestor_sibling_path` can be the name of a sibling directory (or some
    descendant of the sibling) to some ancestor of `path` or `path`
    """
    start = Path(start).absolute()
    if include_start:
        start /= "_"
    for anc in start.parents:
        candidate = anc / subpath
        if candidate.exists(follow_symlinks=not ignore_broken_symlinks):
            return candidate
    raise FileNotFoundError(f"No ancestor of {start} has a child {subpath}.")


def _split_long_name(name, max_length=255):
    result, remainder = [], name
    while len(remainder) > max_length:
        result.append(remainder[:max_length])
        remainder = remainder[max_length:]
    result.append(remainder)
    return result


def _split_long_names(path: Path, max_length=255):
    partses = [_split_long_name(p, max_length) for p in path.parts]
    return functools.reduce(list.__add__, partses, [])


def to_valid_path(path, split_long_names=False, max_name_length=255):
    path = str(path).strip()
    allowed = r"-\w.,\'\\/!#$%^&()_+=@{}\[\]"
    path = Path(re.sub(f"(?u)[^{allowed}]", "+", str(path)))
    if split_long_names:
        parts = _split_long_names(path, max_name_length)
        path = Path(os.path.join(*parts))
    return path


def get_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total_size += os.path.getsize(fp)
            except FileNotFoundError as e:  # the file has just been deleted
                pass
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


# File info ########################################################################################

def time_since_access(file):
    access_time = dt.datetime.utcfromtimestamp(Path(file).stat().st_atime)
    return dt.datetime.utcnow() - access_time


# File #############################################################################################


def create_file_atomic(path, write_action, mode="w+b"):
    """Performs `write_action` on a temporary file and, if successful, moves the
    temporary file to `path`. This avoids creation of incomplete files if an
    error during writing occurs.

    Args:
        path: The path of the output file.
        write_action: A procedure that accepts a file as the only argument.
        mode: File opening mode.
    """
    tmp = tempfile.NamedTemporaryFile(mode=mode, delete=False, dir=Path(path).parent)
    try:
        write_action(tmp.file)
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
    return Path(path).open("w+" if append else "w").write(text)


def write_lines(path, lines, append=False):
    with open(path, "w+" if append else "w") as file:
        file.writelines(lines)


def disk_partition(path):
    # https://stackoverflow.com/questions/25283882/determining-the-filesystem-type-from-a-path-in-python
    import psutil
    best_match = ""
    partition = None
    for part in psutil.disk_partitions():
        if path.startswith(part.mountpoint) and len(best_match) < len(part.mountpoint):
            partition, best_match = part, part.mountpoint
    return partition


def fs_type(path):
    return disk_partition(path).fstype
