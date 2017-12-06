"""Function to initialize a population."""
import numpy as np


def initialize_population(pop_size, d_param):
    """Generate a random starting population.

    In this basic implementation, all parameters are randomly assigned between
    [0., 1.] which only makes sense if the data is on this approximate scale.

    Parameters
    ----------
    pop_size : int
        Population size.
    d_param : int
        Dimension of parameters in model.
    """
    pop = []
    for i in range(pop_size):
        new_param = []
        for j in range(len(d_param)):
            new_param.append(list(np.random.rand(d_param[j])))
        pop.append(new_param)

    return pop
