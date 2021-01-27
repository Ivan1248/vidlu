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


requirements = [r if not r.startswith("pillow") else process_pillow_req(r)
                for r in Path('requirements.txt').read_text().splitlines()]

if os.environ.get("VIDLU_OPTIONAL_REQUIREMENTS", False):
    requirements += Path('requirements-optional.txt').read_text().splitlines()

setup(name="vidlu", packages=find_packages(), install_requires=requirements)
