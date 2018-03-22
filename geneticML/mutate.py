import copy
import numpy as np

def block_mutation(parent_one, mut_op):
    """Perform a random permutation on a parameter block.

    Parameters
    ----------
    parent_one : list
        List of params for first parent.
    mut_op : string
        String of operator for mutation.
    """
    p1 = copy.deepcopy(parent_one)
    p1_index = []
    for i in range(len(p1)):
        p1_index.append([i] * len(p1[i]))

    mut_block = np.random.randint(0, len(p1_index), 1)[0]
    mut_point = np.random.randint(0, len(p1[mut_block]), 1)[0]

    old_params = p1[mut_block][mut_point]
    new_params = np.random.rand(1)[0] * 2. * (old_params)
    if mut_op != '=':
        rparams = eval('old_params ' + mut_op + ' new_params')
    else:
        rparams = new_params
    p1[mut_block][mut_point] = np.abs(rparams)

    return p1
