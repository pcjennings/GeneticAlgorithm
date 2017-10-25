import random

from genetic import GeneticAlgorithm


def ff(x):
    return random.random()


ga = GeneticAlgorithm(pop_size=10,
                      fit_func=ff,
                      d_param=[1, 1, 3],
                      pop=None)
ga.search(5000)

print(ga.pop)
print(ga.fitness)
