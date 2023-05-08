#!/usr/bin/env python
import os
from pathlib import Path

from setuptools import setup

with open("torchbuilder/core/__init__.py") as file:
    for line in file.readlines():
        if "version" in line:
            version = line.split("=")[1].strip().replace('"', "")
            break

assert (
    os.path.exists(os.path.join("torchbuilder", "__init__.py")) is False
), "torchbuilder is a namespace not a module"

extra_requires = {"plugins": ["importlib_resources"]}
extra_requires["all"] = sorted(set(sum(extra_requires.values(), [])))

if __name__ == "__main__":
    setup(
        name="torchbuilder",
        version=version,
        extras_require=extra_requires,
        description="Simple utility to compute shape side while building a nnet",
        long_description=(Path(__file__).parent / "README.rst").read_text(),
        author="setepenre",
        author_email="setepenre@outlook.com",
        license="BSD 3-Clause License",
        url="https://torchbuilder.readthedocs.io",
        classifiers=[
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Operating System :: OS Independent",
        ],
        packages=[
            "torchbuilder.core",
            "torchbuilder.plugins.example",
        ],
        setup_requires=["setuptools"],
        install_requires=["importlib_resources", "torch"],
        namespace_packages=[
            "torchbuilder",
            "torchbuilder.plugins",
        ],
        package_data={
            "torchbuilder.data": [
                "torchbuilder/data",
            ],
        },
    )
