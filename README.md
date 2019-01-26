# Combination-dependent Factorization Machines
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/moriaki3193/cdfm/branch/master/graph/badge.svg)](https://codecov.io/gh/moriaki3193/cdfm)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![PyPI](https://img.shields.io/pypi/v/cdfm.svg)](https://pypi.org/project/cdfm/)

## Install
```
$ pip install cdfm
```

## Development
```shell
# 1. install develop dependencies
$ pip install -e .[dev]

# 2. linting
$ pylint cdfm  # check pylintrc for more details...

# 3. type checking
$ mypy @mypy_check_files --config-file=mypy.ini

# 4. testing
$ pytest
```