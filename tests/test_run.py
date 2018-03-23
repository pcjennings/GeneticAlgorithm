import unittest
import random
import numpy as np

from geneticML.algorithm import GeneticAlgorithm as paramGA
from featGA.algorithm import GeneticAlgorithm as featGA


class TestGeneticAlgorithm(unittest.TestCase):
    def ff(self, x):
        """Some random fitness is returned."""
        return random.random()

    def test_run_ga(self):
        """Simple test case to make sure it doesn't crash."""
        ga = paramGA(pop_size=10,
                     fit_func=self.ff,
                     d_param=[1, 100, 3],
                     pop=None)
        ga.search(500)

        self.assertTrue(len(ga.pop) == 10)
        self.assertTrue(len(ga.fitness) == 10)

    def test_feature_selection(self):
        ga = featGA(pop_size=10,
                    fit_func=self.ff,
                    dimension=20,
                    pop=None)
        self.assertEqual(np.shape(ga.pop), (10, 20))
        print(ga.pop)

        ga.search(500)
        print(ga.pop)


if __name__ == '__main__':
    unittest.main()
