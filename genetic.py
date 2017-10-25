import numpy as np


class GeneticAlgorithm(object):
    """Genetic algorithm for parameter optimization."""

    def __init__(self, pop_size):
        """Initialize the genetic algorithm.

        Parameters
        ----------
        pop_size : int
            Population size.
        """
        self.pop_size = pop_size

    def initialize_population(self, d_param):
        """Generate a random starting population.

        Parameters
        ----------
        d_param : int
            Dimension of parameters in model.
        """
        self.d_param = d_param

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



if __name__ == '__main__':
    ga = GeneticAlgorithm(pop_size=3)
    pop = ga.initialize_population([1, 1, 3])
    print(ga.cut_and_splice(pop[0], pop[1]))
