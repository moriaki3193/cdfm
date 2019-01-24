# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


INSTALL_REQUIRES = ['numpy', 'pandas', 'fastprogress']
KEYWORDS = 'FMs L2R LETOR'
CLASSIFIERS = [
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
]

# Development dependencies
EXTRA_REQUIRES = {
    'dev': ['pytest', 'pytest-cov', 'mypy', 'pylint', 'line_profiler']
}

setup(
    name='cdfm',
    version='0.0.1',
    description='Combination-dependent Factorization Machines',
    author='Moriaki Saigusa',
    author_email='moriaki3193@gmail.com',
    url='https://github.com/moriaki3193/cdfm',
    license=license,
    packages=find_packages(exclude=('tests')),
    install_requires=INSTALL_REQUIRES,
    extra_requires=EXTRA_REQUIRES,
    keywords=KEYWORDS,
    test_suite='tests',
    classifiers=CLASSIFIERS,
    zip_safe=False,
)
