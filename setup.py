#!/usr/bin/env python

"""The setup script."""

import os
import rts_package as module
from setuptools import setup, find_packages


def walker(base, *paths):
    file_list = set([])
    cur_dir = os.path.abspath(os.curdir)

    os.chdir(base)
    try:
        for path in paths:
            for dname, dirs, files in os.walk(path):
                for f in files:
                    file_list.add(os.path.join(dname, f))
    finally:
        os.chdir(cur_dir)

    return list(file_list)


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('CHANGELOG.rst') as history_file:
    history = history_file.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup_requirements = 'pytest-runner'

test_requirements = 'pytest'

setup(
    author="Julian Wanner",
    author_email='jwgithub@mailbox.org',
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description="An mlf-core prediction package for root tissue segmentation.",
    entry_points={
        'console_scripts': [
            'rts-pred=rts_package.cli_pred:main',
            'rts-pred-uncert=rts_package.cli_uncert_pred:main',
            'rts-feat-imp=rts_package.cli_feat_imp:main',
        ],
    },
    install_requires=requirements,
    license="MIT",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='root tissue segmentation',
    name='root-tissue-seg-package',
    packages=find_packages(include=['rts_package', 'rts_package.*']),
    package_data={
        module.__name__: walker(
            os.path.dirname(module.__file__),
            'models', 'data'
        ),
    },
    setup_requires=setup_requirements,
    tests_require=test_requirements,
    url='https://github.com/qbic-pipelines/rts-prediction-package/',
    version='1.0.5',
    zip_safe=False,
)
