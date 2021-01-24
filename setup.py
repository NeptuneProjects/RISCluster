#!/usr/bin/env python3

from setuptools import setup

def setup_package():
    setup(
        name='RISCluster',
        url="'ttps://github.com/NeptuneProjects/RISCluster',
        author='William F. Jenkins II',
        author_email='wjenkins@ucsd.edu',
        packages=['RISCluster'],
        # scripts=["scripts/testscript.py", "scripts/testscript2.py"],
        entry_points = {
            'console_scripts': [
                'cleancat=RISProcess.commands:cleancat',
                'dlfdsn=RISProcess.commands:dlfdsn',
                'process=RISProcess.commands:process',

            ]
        },
        install_requires=[
            'cmocean',
            'h5py',
            'jupyterlab',
            'matplotlib',
            'numpy',
            'obspy',
            'pandas',
            'pydotplus',
            'python-dotenv',
            'torch',
            'torchvision',
            'scikit-learn',
            'scikit-learn-extra',
            'scipy',
            'seaborn',
            'tensorboard',
            'tqdm',
            'twilio'
        ],
        version='0.0b0',
        license='MIT',
        description="Package provides Pytorch implementation of deep embedded \
            clustering for data recorded on the Ross Ice Shelf, Antarctica."
    )


if __name__ == '__main__':
    setup_package()