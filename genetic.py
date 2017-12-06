import numpy as np
from random import shuffle


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

        self.pop = pop
        if self.pop is None:
            self.initialize_population()

    def initialize_population(self):
        """Generate a random starting population."""
        self.pop = []
        for i in range(self.pop_size):
            new_param = []
            for j in range(len(self.d_param)):
                new_param.append(list(np.random.rand(self.d_param[j])))
            self.pop.append(new_param)

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
            cut_point = np.random.randint(1, len(parent_one), 1)[0]
        offspring = parent_one[:cut_point] + parent_two[cut_point:]

        return offspring

    def block_mutation(self, parent_one, mut_op):
        """Perform a random permutation on a parameter block.

        Parameters
        ----------
        parent_one : list
            List of params for first parent.
        mut_op : string
            String of operator for mutation.
        """
        p1 = parent_one.copy()
        mut_point = np.random.randint(0, len(p1), 1)[0]
        old_params = np.array(p1[mut_point])
        new_params = np.random.rand(len(p1[mut_point])) * 2. * (old_params)
        if mut_op != '=':
            rparams = eval('old_params ' + mut_op + ' new_params')
        else:
            rparams = new_params
        p1[mut_point] = list(np.abs(rparams))

        return p1

    def get_fitness(self, param_list):
        """Function wrapper to calculate the fitness.

        Parameters
        ----------
        param_list : list
            List of new parameter sets to get fitness for.
        """
        fit = []
        for p in param_list:
            try:
                calc_fit = self.fit_func(p)
            except ValueError:
                calc_fit = float('-inf')

            fit.append(calc_fit)

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
        index = list(range(len(fit_list)))
        fit = list(zip(*sorted(zip(fit_list, index), reverse=True)))

        scale = []
        s = 0
        for i in fit[1]:
            s += 1 / (len(fit[1]) + 2)
            scale.append(s)

        fit_list = list(zip(*sorted(zip(fit[1], scale), reverse=False)))[1]

        param_list_shuf = []
        fit_list_shuf = []
        index_shuf = list(range(len(param_list)))
        shuffle(index_shuf)
        for i in index_shuf:
            param_list_shuf.append(param_list[i])
            fit_list_shuf.append(fit_list[i])

        param_list, fit_list = param_list_shuf, fit_list_shuf

        # Get random probability.
        for i, j in zip(param_list, fit_list):
            if j > np.random.rand(1)[0]:
                return i

    def population_reduction(self, pop, fit):
        """Method to reduce population size to constant.

        Parameters
        ----------
        pop : list
            Extended population.
        fit : list
            Extended fitness assignment.
        """
        global_details = [[i, j] for i, j in zip(pop, fit)]
        global_details.sort(key=lambda x: float(x[1]), reverse=True)

        self.pop, self.fitness, unique_list = [], [], []
        for i in global_details:
            if len(self.pop) < self.pop_size:
                if round(i[1], 2) not in unique_list:
                    self.pop.append(i[0])
                    self.fitness.append(i[1])
                    unique_list.append(round(i[1], 2))
            else:
                break

        assert len(self.pop) == len(self.fitness)

    def search(self, steps):
        """Do the actual search.

        Parameters
        ----------
        steps : int
            Maximum number of steps to be taken.
        """
        self.fitness = self.get_fitness(self.pop)
        operator = [self.cut_and_splice, self.block_mutation]
        # base_mut_op = ['=', '+', '-', '/', '**', '** -1. *', '** 0.5 *',
        #               '/10.*', '/100.*', '/1000.*', '*2.*', '*5.*', '*10.*']
        base_mut_op = ['=', '+', '-', '/', '*']

        for _ in range(steps):
            p = self.pop[0][0][0]
            offspring_list = []
            for c in range(self.pop_size):
                op = np.random.randint(0, len(operator), 1)[0]
                p1 = None
                while p1 is None:
                    p1 = self.selection(self.pop, self.fitness)
                    assert self.pop[0][0][0] == p
                if op == 0:
                    op = operator[op]
                    p2 = p1
                    while p2 is p1 or p2 is None:
                        p2 = self.selection(self.pop, self.fitness)
                        assert self.pop[0][0][0] == p
                    offspring_list.append(op(p1, p2))
                    assert self.pop[0][0][0] == p
                else:
                    mut_choice = np.random.randint(0, len(base_mut_op), 1)[0]
                    assert self.pop[0][0][0] == p
                    op = operator[op]
                    offspring_list.append(
                        op(p1, mut_op=base_mut_op[mut_choice])
                        )
                    assert self.pop[0][0][0] == p
            new_fit = self.get_fitness(offspring_list)

            if new_fit is None:
                break

            extend_fit = self.fitness + new_fit
            extend_pop = self.pop + offspring_list
            self.population_reduction(extend_pop, extend_fit)
