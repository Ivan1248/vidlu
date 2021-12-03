from setuptools import setup, find_packages
from pkg_resources import get_distribution, DistributionNotFound
from pathlib import Path
import os


def get_dist(pkgname):
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None


def process_pillow_req(pillow_req):
    pillow_ver = (pillow_req.split() + [""])[1]
    pillow_req = 'pillow-simd' if get_dist('pillow-simd') is not None else 'pillow'
    return pillow_req + " " + pillow_ver


def parse_requirements(text):
    requirements = []
    dependency_links = []
    DEP_PREFIX = "--find-links"
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(DEP_PREFIX):
            dependency_links.append(line[len(DEP_PREFIX):].strip())
        elif line.startswith("pillow"):
            requirements.append(process_pillow_req(line))
        else:
            requirements.append(line)
    return requirements, dependency_links


requirements, dependency_links = parse_requirements(Path('requirements.txt').read_text())

requirements_optional = Path('requirements-optional.txt').read_text().splitlines()

version = {}
with open("vidlu/version.py") as fp:
    exec(fp.read(), version)

setup(name="vidlu", version=version['__version__'], packages=find_packages(),
      install_requires=requirements, dependency_links=dependency_links,
      extras_require={'full': requirements_optional})
