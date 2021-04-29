from setuptools import setup
from setuptools import find_packages

exclude_dirs = ("configs",)

# for install, do: pip install -ve .

setup(
    name='FusionTransformer',
    version="0.0.1",
    url="https://github.com/aliabdelkader/FusionTransformer",
    description="Transformer and Lidar fusion for semantic segmentation",
    install_requires=['yacs', 'nuscenes-devkit', 'tabulate', 'timm', 'dataclasses', 'tensorboard', 'PyYAML'],
    packages=find_packages(exclude=exclude_dirs),
)