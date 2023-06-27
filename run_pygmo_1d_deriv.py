import numpy as np
import pygmo as pg
import time

from configs.parametersOneD import ParametersOneD_cft, Parameters_pygmo
from environment.cfts import CrossingOneD_deriv
import environment.pygmo_udps as udps
import environment.utils as utils

# uncomment line below for deterministic testing
# pg.set_global_rng_seed(seed=27021976)


def reorder_solution_1d(solution):
    """
    Applies a custom sorting of the cft data.

    The first 10 conformal weights (i.e. all J=1,2,3 operators and the first J=4 operator)
    and corresponding ope coefficients are considered fixed.
    The remaining, non-fixed, cft data is then split into degeneracies according to spin J values.
    Each degenerate family is sorted in increasing order of conformal dimension
    and reassembled into a single array again.

    Parameters
    ----------
    solution : array_like
               The array of cft data to be sorted.

    Returns
    -------
    sorted_solution : array_like
                    The sorted cft data.

    """
    deltas = solution[:solution.size // 2]
    opes = solution[solution.size // 2:]
    non_fixed_deltas = deltas[10:]
    non_fixed_opes = opes[10:]
    partition = 1 + np.where(np.diff(params.spin_list_long[10:]) != 0)[0]
    zip_x = list(zip(non_fixed_deltas, non_fixed_opes))
    zip_x_split_by_spins = np.split(zip_x, partition)
    sorted_splits = [sorted(array.tolist()) for array in zip_x_split_by_spins]
    sorted_deltas, sorted_opecoeffs = np.split(np.concatenate(sorted_splits), 2, axis=1)
    sorted_solution = np.concatenate((deltas[:10], sorted_deltas[:, 0], opes[:10], sorted_opecoeffs[:, 0]))
    return sorted_solution


if __name__ == '__main__':
    # ---Instantiating relevant parameters classes---
    params = ParametersOneD_cft()
    params_pg = Parameters_pygmo()

    start_time = time.time()

    # ---Instantiate the crossing_eqn class---
    cft = CrossingOneD_deriv(params)

    # get user defined problem
    udp = getattr(udps, params_pg.udp)

    if params_pg.udp == 'udp_1d_integral_constraints':
        # ---Load the pre-generated integrated constraint data---
        suffixes = [str(i) + '_to_' + str(i + 1) for i in range(params.delta_end)]
        int1_list = utils.generate_block_list_csv('1d constraints/1D_Int1_delta_', suffixes, [])
        int2_list = utils.generate_block_list_csv('1d constraints/1D_Int2_delta_', suffixes, [])
        super_int1 = np.hstack([block for block in int1_list])
        super_int2 = np.hstack([block for block in int2_list])
        rhs1 = np.loadtxt('block_lattices/1d constraints/Rhs1.csv', delimiter=',')
        rhs2 = np.loadtxt('block_lattices/1d constraints/Rhs2.csv', delimiter=',')

        prob = pg.problem(udp(cft, super_int1, rhs1, super_int2, rhs2))
    else:
        prob = pg.problem(udp(cft))

    # get the pygmo algorithm to use
    user_algo = getattr(pg, params_pg.pygmo_algo_name)
    algo = pg.algorithm(user_algo(**params_pg.pygmo_algo_dict))
    algo.set_verbosity(params_pg.verbosity)
    pop = pg.population(prob, size=params_pg.population_size)
    # get the random seed used in the population so that we can feed it into
    # other algorithms to directly compare results
    pop_seed = pop.get_seed()
    pop = algo.evolve(pop)
    sorted_x = reorder_solution_1d(pop.champion_x)
    out = np.concatenate((round(cft.coupling, 4), pop.champion_f, sorted_x), axis=None)
    utils.output_to_file(params_pg.output_file, out)
    del algo, pop

    print("--- %s seconds ---" % (time.time() - start_time))
