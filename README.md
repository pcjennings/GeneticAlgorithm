# Genetic Algorithm

[![Build Status](https://travis-ci.org/pcjennings/GeneticAlgorithm.svg?branch=master)](https://travis-ci.org/pcjennings/GeneticAlgorithm)
<a href="https://codeclimate.com/github/pcjennings/GeneticAlgorithm/maintainability"><img src="https://api.codeclimate.com/v1/badges/defb74010402e415f9db/maintainability" /></a>
[![codecov](https://codecov.io/gh/pcjennings/GeneticAlgorithm/branch/master/graph/badge.svg)](https://codecov.io/gh/pcjennings/GeneticAlgorithm)

Experimental genetic algorithm for parameter optimization.

Performs global optimization of hyperparameters based on the log marginal
likelihood as a measure of fitness. Currently works with a private code. This
should become open source soon.

## Table of contents

-   [Installation](#installation)
-   [Usage](#usage)
-   [Authors](#authors)

## Installation
[(Back to top)](#table-of-contents)

The easiest way to install the code is with:

  ```shell
    pip install git+https://gitlab.com/atoML/AtoML.git
  ```

This will automatically install the code as well as the dependencies.
Alternatively, you can clone the repository to a local directory with:

  ```shell
    git clone https://gitlab.com/atoML/AtoML.git
  ```

And then put the `<install_dir>/` into your `$PYTHONPATH` environment variable.

Be sure to install dependencies in with:

  ```shell
    pip install -r requirements.txt
  ```

#### Requirements

numpy

## Usage
[(Back to top)](#table-of-contents)

In the most basic form, it is possible to set up a search using the following
lines of code:

```python
  # Setup the GA search.
  ga = GeneticAlgorithm(pop_size=50,
                        fit_func=fitness_func,
                        d_param=[5, 3],
                        pop=None)
  # Run GA search.
  ga.search(500)
```

In this case, the `fitness_func` can be any user defined fitness function. The
`d_param` variable gives the dimensionality of the model parameters to be
optimized.

## Authors
[(Back to top)](#table-of-contents)

-   [Paul Jennings](http://suncat.stanford.edu/theory/people/paul-jennings)
