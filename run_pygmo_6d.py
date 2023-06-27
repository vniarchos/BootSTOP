import numpy as np
import pygmo as pg
import time

from configs.parametersSixD import ParametersSixD_cft, Parameters_pygmo
from environment.cfts import CrossingSixD
from environment.data_z_sample import ZData_6d_250
import environment.pygmo_udps as udps
import environment.utils as utils

# uncomment line below for deterministic testing
# pg.set_global_rng_seed(seed=27021976)


if __name__ == '__main__':
    # ---Instantiating relevant parameters classes---
    params = ParametersSixD_cft()
    params_pg = Parameters_pygmo()

    # ---Kill portion of the z-sample data if required---
    zd = ZData_6d_250()
    zd.kill_data(params.z_kill_list)

    start_time = time.time()

    # ---Instantiate the crossing_eqn class---
    cft = CrossingSixD(params, zd)

    # get user defined problem
    udp = getattr(udps, params_pg.udp)
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
    sorted_x = pop.champion_x  # don't sort for 6d
    out = np.concatenate((round(cft.coupling, 4), pop.champion_f, sorted_x), axis=None)
    utils.output_to_file(params_pg.output_file, out)
    del algo, pop

    print("--- %s seconds ---" % (time.time() - start_time))
