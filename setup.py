#!/usr/bin/env python3

from setuptools import setup

def setup_package():
    setup(
        name='RISCluster',
        url='https://github.com/NeptuneProjects/RISCluster',
        author='William F. Jenkins II',
        author_email='wjenkins@ucsd.edu',
        packages=['RISCluster'],
        scripts=['RISCluster/runDC'],
        entry_points = {
            'console_scripts': [
                'query_H5size=RISCluster.utils:query_H5size',
                'extract_H5dataset=RISCluster.utils:extractH5dataset',
                'generate_sample_index=RISCluster.utils:generate_sample_index',
                'convert_H5_to_NP=RISCluster.utils:convert_H5_to_NP'
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
            'scipy',
            'tensorboard',
            'tqdm'
        ],
        version='0.3',
        license='MIT',
        description="Package provides Pytorch implementation of deep embedded \
            clustering for data recorded on the Ross Ice Shelf, Antarctica."
    )


if __name__ == '__main__':
    setup_package()
