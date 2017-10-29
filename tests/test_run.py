import unittest
import random

from genetic import GeneticAlgorithm


class TestGeneticAlgorithm(unittest.TestCase):
    def ff(self, x):
        """Some random fitness is returned."""
        return random.random()

    def test_run_ga(self):
        """Simple test case to make sure it doesn't crash."""
        ga = GeneticAlgorithm(pop_size=10,
                              fit_func=self.ff,
                              d_param=[1, 100, 3],
                              pop=None)
        ga.search(500)

        self.assertTrue(len(ga.pop) == 10)
        self.assertFalse(len(ga.fitness) == 10)


if __name__ == '__main__':
    unittest.main()
