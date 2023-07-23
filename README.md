# Ocean

## Easy ML workflows built on scikit-learn

Ocean is a package for machine learning workflows that makes it easier to create models. It is made using several tools from scikit-learn, to speed up the machine learning process.

## Feature

This package will continue to be developed to add more features, but for now there is only one feature, namely automl.

- automl

## Dependency

```console
numpy==1.25.1
scikit-learn==1.3.0
scipy==1.11.1
tabulate==0.9.0
xgboost==1.7.6
```

## How to Install

You can install ocean with pip using this command:
```console
pip install ocean-ml
```

## Example

#### AutoML

```python
from ocean.automl import AutoMLRegressor

model = AutoMLRegressor(X, y, categoric, tune_params=True)
model.fit()
```

## Pypi Link

https://pypi.org/project/ocean-ml/

## Github Link
https://github.com/AgufSamudra/ocean <br>
[Ocean](https://github.com/AgufSamudra/ocean)
