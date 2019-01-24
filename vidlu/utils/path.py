from pathlib import Path


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
