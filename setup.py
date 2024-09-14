"""Installation script for the 'isaacgymenvs' python package."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from setuptools import setup, find_packages

import os

root_dir = os.path.dirname(os.path.realpath(__file__))


# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # RL
    "gym==0.23.1",
    "torch",
    "omegaconf",
    "termcolor",
    "jinja2",
    "hydra-core>=1.1",
    "rl-games>=1.6.0",
    "pyvirtualdisplay",
    "moviepy",
    "stable_baselines==2.0.0",
    "gymnasium_robotics[mujoco]",
    "ml_collections",
    "cython<3"
    "gymnasium==0.28.1",
    "tensorboardx==2.6.2.2",
    ]


# Installation operation
setup(
    name="isaacgymenvs",
    author="NVIDIA",
    version="1.4.0",
    description="Benchmark environments for high-speed robot learning in NVIDIA IsaacGym.",
    keywords=["robotics", "rl"],
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages("."),
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.6, 3.7, 3.8, 3.9"],
    zip_safe=False,
)

# EOF
