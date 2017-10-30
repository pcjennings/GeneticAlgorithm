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
        mut_point = np.random.randint(0, len(parent_one), 1)[0]
        old_params = np.array(parent_one[mut_point])
        new_params = np.random.rand(len(parent_one[mut_point]))
        if mut_op != '=':
            rparams = eval('old_params ' + mut_op + ' new_params')
        else:
            rparams = new_params
        parent_one[mut_point] = list(np.abs(rparams))

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
        global_details.sort(key=lambda x: float(x[1]), reverse=False)

        self.pop, self.fitness = [], []
        for i in global_details:
            if len(self.pop) < self.pop_size:
                self.pop.append(i[0])
                self.fitness.append(i[1])
            else:
                break

    def search(self, steps):
        """Do the actual search.

        Parameters
        ----------
        steps : int
            Maximum number of steps to be taken.
        """
        self.fitness = self.get_fitness(self.pop)
        operator = [self.cut_and_splice, self.block_mutation]
        base_mut_op = ['=', '+', '-', '/', '**', '** -1. *']

        for s in range(steps):
            offspring_list = []
            for c in range(self.pop_size):
                op = np.random.randint(0, len(operator), 1)[0]
                p1 = None
                while p1 is None:
                    p1 = self.selection(self.pop, self.fitness)
                if op == 0:
                    op = operator[op]
                    p2 = p1
                    while p1 is p2 and None:
                        p2 = self.selection(self.pop, self.fitness)
                    offspring_list.append(op(p1, p2))
                else:
                    mut_choice = np.random.randint(0, len(base_mut_op), 1)[0]
                    op = operator[op]
                    offspring_list.append(op(p1,
                                             mut_op=base_mut_op[mut_choice]))
            extend_fit = self.fitness + self.get_fitness(offspring_list)
            extend_pop = self.pop + offspring_list
            self.population_reduction(extend_pop, extend_fit)
