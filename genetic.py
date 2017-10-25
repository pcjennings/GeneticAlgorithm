import numpy as np


class GeneticAlgorithm(object):
    """Genetic algorithm for parameter optimization."""

    def __init__(self, pop_size, fit_func, d_param, pop=None):
        """Initialize the genetic algorithm.

        Parameters
        ----------
        pop_size : int
            Population size.
        fit_func : object
            User defined function to calculate fitness.
        d_param : int
            Dimension of parameters in model.
        pop : list
            The current population. Default is None.
        """
        self.pop_size = pop_size
        self.fit_func = fit_func
        self.d_param = d_param

        if pop is None:
            pop = self.initialize_population()
        self.pop = pop

    def initialize_population(self):
        """Generate a random starting population."""
        self.pop = []
        for i in range(self.pop_size):
            new_param = []
            for j in range(len(self.d_param)):
                new_param.append(list(np.random.rand(self.d_param[j])))
            self.pop.append(new_param)

        return self.pop

    def cut_and_splice(self, parent_one, parent_two, size='random'):
        """Perform cut_and_splice between two parents.

        Parameters
        ----------
        parent_one : list
            List of params for first parent.
        parent_two : list
            List of params for second parent.
        size : str
            Define how to choose size of each cut.
        """
        if size is 'random':
            cut_point = np.random.randint(1, len(parent_one)+1, 1)[0]
        child = parent_one[:cut_point] + parent_two[cut_point:]

        return child

    def block_mutation(self, parent_one):
        """Perform a random permutation on a parameter block.

        Parameters
        ----------
        parent_one : list
            List of params for first parent.
        """
        mut_point = np.random.randint(0, len(parent_one), 1)[0]
        new_params = list(np.random.rand(len(parent_one[mut_point])))
        parent_one[mut_point] = new_params

        return parent_one

    def get_fitness(self, param_list):
        """Function wrapper to calculate the fitness.

        Parameters
        ----------
        param_list : list
            List of new parameter sets to get fitness for.
        """
        fit = []
        for p in param_list:
            fit.append(self.fit_func(p))

        return fit

    def selection(self, param_list, fit_list):
        """Perform natural selection.

        Parameters
        ----------
        param_list : list
            List of parameter sets to consider.
        fit_list : list
            list of fitnesses associated with parameter list.
        """
        fit_list = np.asarray(fit_list)
        # Scale the current set of fitnesses.
        fit_list = (fit_list - np.min(fit_list)) / np.max(fit_list)
        # Get random probability.
        prob = np.random.rand(len(fit_list))
        for i, j, k in zip(param_list, fit_list, prob):
            if j > k:
                return i


if __name__ == '__main__':
    ga = GeneticAlgorithm(pop_size=3)
    pop = ga.initialize_population([1, 1, 3])
    child = ga.cut_and_splice(pop[0], pop[1])
    child = ga.block_mutation(pop[0])
