from setuptools import setup, find_packages
from pathlib import Path
import os

requirements = Path('requirements.txt').read_text().splitlines()

if os.environ.get("VIDLU_OPTIONAL_REQUIREMENTS", False):
    requirements += Path('requirements-optional.txt').read_text().splitlines()

setup(name="vidlu", packages=find_packages(), install_requires=requirements)
