# auto-sklearn

This is forked from automl/auto-sklearn of v0.4.0 for study and development purpose. 
See Modification section for more details.

auto-sklearn is an automated machine learning toolkit and a drop-in replacement for a scikit-learn estimator.

Find the documentation [here](http://automl.github.io/auto-sklearn/)

Status for master branch

[![Build Status](https://travis-ci.org/automl/auto-sklearn.svg?branch=master)](https://travis-ci.org/automl/auto-sklearn)
[![Code Health](https://landscape.io/github/automl/auto-sklearn/master/landscape.png)](https://landscape.io/github/automl/auto-sklearn/master)
[![codecov](https://codecov.io/gh/automl/auto-sklearn/branch/master/graph/badge.svg)](https://codecov.io/gh/automl/auto-sklearn)

Status for development branch

[![Build Status](https://travis-ci.org/automl/auto-sklearn.svg?branch=development)](https://travis-ci.org/automl/auto-sklearn)
[![Code Health](https://landscape.io/github/automl/auto-sklearn/development/landscape.png)](https://landscape.io/github/automl/auto-sklearn/development)
[![codecov](https://codecov.io/gh/automl/auto-sklearn/branch/development/graph/badge.svg)](https://codecov.io/gh/automl/auto-sklearn)

# Modification

This is a branch of auto-sklearn for study and development purpose. 
The ideas are provided by Dr. Tom Cheng and implemented by myself.
Only raw results will be shown until the paper come out.

Following files have been modified:

### data/

* dataset_1049_pc4.csv <br>
We use pc4, real-sim and rcv1 for testing.

### results/

* fitting_loss_in_smbo/ <br>
Excel file for the comparison of auto-sklearn with/without
positive direction.