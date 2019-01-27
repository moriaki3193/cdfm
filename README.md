# Combination-dependent Factorization Machines
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/moriaki3193/cdfm/branch/master/graph/badge.svg)](https://codecov.io/gh/moriaki3193/cdfm)
[![PyPI](https://img.shields.io/pypi/v/cdfm.svg)](https://pypi.org/project/cdfm/)
<!--
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
-->

## requirements
- CPython 3.6.x, 3.7.x

## dependencies
- [NumPy](http://www.numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [fastprogress](https://github.com/fastai/fastprogress)

## install
```
$ pip install cdfm
```

## usage
### 1. prepare your dataset
The dataset format is like SVM-rank one.
The difference is `eid` must be specified in a line.
Here is a definition of a line.
`|` symbol means `OR` (so `<str>|<int>` means the value must have either str or int type).

#### fundamentals
```txt
<line>     .=. <label> qid:<qid> eid:<eid> <features>#<comments>

<label>    .=. <float>|<str as a class>
<qid>      .=. <str>|<int>
<eid>      .=. <str>|<int>
<features> .=. <dim>:<value>
<dim>      .=. <0 or Natural Number>
<value>    .=. <float>
<comments> .=. <Any text will do>
```

Let me show you an example.

```txt
0.5 qid:1 eid:x 1:0.1 2:-0.2 3:0.3 # comment A
0.0 qid:1 eid:y 1:-0.1 2:0.2 4:0.4
-0.5 qid:1 eid:z 2:-0.2 3:0.3 4:-0.4 # comment C
0.5 qid:2 eid:y 1:0.1 2:-0.2 3:0.3
0.0 qid:2 eid:z 1:-0.1 2:0.2 4:0.4
-0.5 qid:2 eid:w 2:-0.2 3:0.3 4:-0.4 # comment E
```

#### distance factors
Additionally, you can use distance between entities in a group.
```txt
<line>     .=. qid:<qid> eid:<eid> cid:<cid> <factors> # <comments>

<cid>      .=. <str>|<int>
<factors>  .=. <dim>:<value>
<div>      .=. <0 or Natural Number>
<value>    .=. <float>
<comments> .=. <Any text will do>
```

Let me show you an example.

```txt
qid:3 eid:x cid:y 1:0.5 2:-0.3 3:1.2 # comment A
qid:3 eid:x cid:z 1:0.0 2:0.2 3:0.8 # comment B
qid:3 eid:y cid:z 1:0.2 2:0.3 3:-0.7 # comment C
```

### 2. loading your dataset
```python
from cdfm.utils import load_cdfmdata

# loading dataset as a DataFrame object
# 1. features
features_path = '/path/to/features'
n_dimensions = 10
features = load_cdfmdata(features_path, n_dimensions)
# features.columns
# >>> Index(['label', 'qid', 'eid', 'features'], dtype='object')

# 2. proximities
proximities_path = '/path/to/proximities'
n_dimensions = 2
proximities = load_cdfmdata(proximities_path, n_dimensions, mode='proximity')
# proximities.columns
# >>> Index(['qid', 'eid', 'cid', 'proximities'], dtype='object')

# some preprocessing here...

# Finally, build a dataset
train = build_cdfmdata(features)  # using features only
train = build_cdfmdata(features, proximities)  # using proximities
```

### 3. fitting the model
```python
from cdfm.models import CDFMRanker

# define your model
model = CDFMRanker(k=8, n_iter=300, init_eta=1e-2)
# fitting, printing out epoch losses if verbose is True
model.fit(train, verbose=True)
```

### 4. save the model
```python
import pickle

with open('/path/to/file.pkl', mode='wb') as fp:
    pickle.dump(model, fp)
```

### 5. make prediction
```python
# loading test dataset
test_df = load_cdfmdata(test_path, n_dimensions)
test = build_cdfmdata(test_df)
pred = model.predict(test)
```

## development
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