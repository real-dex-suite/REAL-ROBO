from setuptools import setup, find_packages

setup(
    name='real_robo',
    version='2.0.0',
    packages=find_packages(),
    description='An integrated framework for robotic teleoperation and control',
    url='https://github.com/real-dex-suite/REAL-ROBO',
    author='Jinzhou Li, Hongwei Fan',
    author_email='kingchou007@gmail.com',
    install_requires=[
        'termcolor',
        'pyyaml',
        'h5py',
        'numpy',
        'protobuf',
        'pyzmq',
        'pybullet',
    ],
)