import numpy as np


class ParametersOneD_cft:
    """
    Class used to hold parameters needed to initialise
    CrossingOneD or CrossingOneD_deriv located in environment.cfts.py.

    Attributes
    __________
    g : float
    spin_list_long : ndarray
        A 1-D array of integer operator spins.
        This determines the number of operators in the truncated spectrum.
    same_spin_hierarchy : bool
        A value of True imposes a minimum separation in conformal weights for spin degenerate operators.
        Should be set to False for pygmo runs but can be True or False for SAC runs.
    dyn_shifts : ndarray
        A 1-D array of values for the minimum separation of adjacent spin degenerate operators.
        Only used if `same_spin_hierarchy` is True.
    delta_start : float
        The value of the lowest conformal weight used to pre-generate the conformal block data.
        Controls which pre-generated blocks are loaded.
    delta_sep : float
        The increase in conformal weights between adjacent
        lattice points in the pre-generated conformal block data.
    delta_end : float
        The value of the highest conformal weight used to pre-generate the conformal block data.
        Controls which pre-generated blocks are loaded.
    z_kill_list : list
        A list of positions to delete from the pre-generated conformal block data arrays.
        If a pre-generated conformal block data array has shape (M,N) then
        specifying a non-empty `z_kill_list` will lead to an array of shape
        (M, N - len(z_kill_list)).
    lb_deltas : ndarray
        A 1-D array holding strict lower bounds for the conformal weights.
    ub_deltas : ndarray
        A 1-D array holding strict upper bounds for the conformal weights.
        Practically, no value should be larger than `delta_start` + `delta_end` to ensure
        staying within the range accessible in the pre-generated conformal block data.
    lb_opecoeffs : ndarray
        A 1-D array holding strict lower bounds for the OPE coefficients.
    ub_opecoeffs : ndarray
        A 1-D array holding strict upper bounds for the OPE coefficients.
    include_tail : bool
        A value of True will add the analytic tail contribution to the crossing equation.

    """

    def __init__(self):
        # ---CFT coupling constant---
        self.g = 0.4  # This is the value of coupling

        # ---Spin partition---
        # 'spins' HAVE to be given in ascending order. 'Spin' refers to J -- it is NOT an actual spin
        self.spin_list_long = np.array([1,
                                        2, 2,
                                        3, 3, 3, 3, 3, 3,
                                        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                        5, 5, 5, 5, 5, 5, 5, 5,
                                        6, 6, 6, 6, 6, 6, 6, 6,
                                        7, 7, 7, 7, 7,
                                        8, 8, 8, 8,
                                        9, 9, 9,
                                        10, 10, 10])

        # ---Spin Hierarchy Parameters---
        self.same_spin_hierarchy = False  # should be set to False for use with pygmo
        self.dyn_shifts = 0.00001 * np.array([0,
                                              0, 0,
                                              0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                                              0, 0, 0, 1, 1, 1, 1, 1,
                                              0, 0, 0, 0, 1, 1, 1, 1,
                                              1, 1, 1, 1, 1,
                                              1, 1, 1, 1,
                                              1, 1, 1,
                                              1, 1, 1])  # set the gap between multiplet same spin deltas

        # ---Pre-generated conformal block lattice Parameters---
        self.delta_start = 0  # lower bound
        self.delta_sep = 0.0001  # jump in weights between each lattice point
        self.delta_end = 20  # upper bound
        self.z_kill_list = [list(range(130, 350))]

        # ---Bounds for CFT data---
        self.lb_deltas = np.array([1.1360024526900692,
                                   2.111484971353163, 2.241489897361551,
                                   3.085668898914247, 3.179538060412292, 3.2106154258526653,
                                   3.3150208347668175, 3.323455424785775, 3.4062798732891646,
                                   4.0697186685568685, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                   4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                   4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                   5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
                                   6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                                   7.0, 7.0, 7.0, 7.0, 7.0,
                                   8.0, 8.0, 8.0, 8.0,
                                   9.0, 9.0, 9.0,
                                   10.0, 10.0, 10.0])
        self.ub_deltas = np.array([1.1360024526900891,
                                   2.1114849713531836, 2.2414898973615713,
                                   3.0856688989142675, 3.1795380604123125, 3.210615425852686,
                                   3.315020834766838, 3.3234554247857955, 3.406279873289185,
                                   4.069718668556888, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
                                   5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
                                   5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
                                   6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                                   7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0,
                                   8.0, 8.0, 8.0, 8.0, 8.0,
                                   9.0, 9.0, 9.0, 9.0,
                                   10.0, 10.0, 10.0,
                                   11.0, 11.0, 11.0])
        self.lb_opecoeffs = np.array([0,
                                      0, 0,
                                      0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0,
                                      0, 0, 0, 0,
                                      0, 0, 0,
                                      0, 0, 0])
        self.ub_opecoeffs = np.array([0.2,
                                      0.2, 0.2,
                                      0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                                      0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                                      0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                                      0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                                      0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                                      0.2, 0.2, 0.2, 0.2, 0.2,
                                      0.2, 0.2, 0.2, 0.2,
                                      0.2, 0.2, 0.2,
                                      0.2, 0.2, 0.2])

        # ---Analytical tail---
        self.include_tail = True


class ParametersOneD_sac:
    """Class used to hold the parameters required to configure the soft-Actor-Critic (SAC) algorithm.

    Attributes
    ----------
    filename_stem: str
        Stem of the filename where output is to be saved (.csv is appended automatically in code).
    verbose : {'', 'e', 'o'}
        When the soft-Actor-Critic algorithm should print to the console:
        - ``: no output.
        - `e`: print everytime `reward` is recalculated.
        - `o`: only when `faff_max` is reached and a re-initialisation occurs.
    output_order : {'', 'Reversed'}
        The order to arrange the data saved to csv file
        - ``: default order is (index_array, reward, cft data)
        - `Reversed` - order is (cft data, reward, index_array)
    faff_max : int
        Maximum number of steps spent searching for an improved reward without success.
        When `faff_max` is reached a re-initialisation occurs.
        Higher value means SAC algorithm spends more time searching for a better solution.
    pc_max : int
        Maximum number of re-initialisations before window size decrease.
        Higher value means SAC algorithm spends more time searching for a better solution.
    window_rate : float
        Search window size decrease rate. Range is (0, 1).
        The window_rate multiplies the search window sizes so small values focus the search quickly.
    max_window_exp : int
        Maximum number of search window size decreases.
        The final window sizes will be equal to ( window_rate ** max_window_exp ) * guess_sizes.
    guessing_run_list_deltas : ndarray
        Boolean array which controls the guessing mode status for each conformal weight datum.
        0 = non-guessing mode
        1 = guessing mode.
    guessing_run_list_opes : ndarray
        Boolean array which controls the guessing mode status for each OPE-squared coefficient datum.
        0 = non-guessing mode
        1 = guessing mode
    guess_sizes_deltas : ndarray
        Initial size of search windows for the conformal weights. They need not all be the same value.
        There is an implicit upper bound set by the highest weight in the pregenerated conformal block files.
    guess_sizes_opes : ndarray
        Initial size of search windows for the OPE-squared coefficients. They need not all be the same value.
    global_reward_start : float
        Initial reward value for the SAC algorithm to try and beat.
    global_best : ndarray
        Initial solution for the SAC algorithm to try and improve upon.

    """

    def __init__(self):
        # ---Output Parameters---
        self.filename_stem = '1d_sac'
        self.verbose = 'e'
        self.output_order = ''
        self.faff_max = 300
        self.pc_max = 6
        self.window_rate = 0.6
        self.max_window_exp = 30
        self.guessing_run_list_deltas = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
        self.guessing_run_list_opes = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
        self.guess_sizes_deltas = np.array([1,
                                            1, 1,
                                            1, 1, 1, 1, 1, 1,
                                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                            1, 1, 1, 1, 1, 1, 1, 1,
                                            1, 1, 1, 1, 1, 1, 1, 1,
                                            1, 1, 1, 1, 1,
                                            1, 1, 1, 1,
                                            1, 1, 1,
                                            1, 1, 1])
        self.guess_sizes_opes = np.array([1,
                                          1, 1,
                                          1, 1, 1, 1, 1, 1,
                                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                          1, 1, 1, 1, 1, 1, 1, 1,
                                          1, 1, 1, 1, 1, 1, 1, 1,
                                          1, 1, 1, 1, 1,
                                          1, 1, 1, 1,
                                          1, 1, 1,
                                          1, 1, 1])
        self.global_reward_start = 0.0
        self.global_best = np.array([1.0002,
                                     2, 2,
                                     3, 3, 3, 3, 3, 3,
                                     4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                     5, 5, 5, 5, 5, 5, 5, 5,
                                     6, 6, 6, 6, 6, 6, 6, 6,
                                     7, 7, 7, 7, 7,
                                     8, 8, 8, 8,
                                     9, 9, 9,
                                     10, 10, 10,
                                     0,
                                     0, 0,
                                     0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0,
                                     0, 0, 0, 0,
                                     0, 0, 0,
                                     0, 0, 0
                                     ])


class Parameters_pygmo:
    """Class used to hold parameters needed to initialise and customise a pygmo run."""

    def __init__(self):
        """
        udp : {'udp_basic', 'udp_1d', 'udp_1d_integral_constraints'}
            String holding the name of the user defined problem.
        pygmo_algo_name : str
            String holding the name of the pygmo algorithm to be used.
            Must match the name of an algorithm in the pygmo module.
            See https://esa.github.io/pygmo2/algorithms.html for list,
            examples include 'ipopt', 'de' and 'simulated_annealing'.
        pygmo_algo_dict : dict
            A dictionary used to override the default parameters of the pygmo algorithm.
            See https://esa.github.io/pygmo2/algorithms.html for algorithm specific keyword arguments.
        population_size : int
            The size of the pygmo population to be used.
        verbosity : int
            A verbosity larger than 0 will produce a log with one entry each verbosity fitness evaluations.
        output_file : str
            The filename of the file where the result of the algorithm will be saved.

        """

        self.udp = 'udp_basic'
        self.pygmo_algo_name = 'ipopt'
        self.pygmo_algo_dict = dict()

        # Example Pygmo algorithm parameter dictionaries
        # self.dict_de = dict(gen=10, F=0.8, CR=0.9, variant=2, ftol=1e-6, xtol=1e-6, seed=random)
        #
        # self.dict_sim_ann = dict(Ts=10., Tf=0.1, n_T_adj=10, n_range_adj=10, bin_size=10, start_range=1., seed=random)
        #
        # self.dict_pso = dict(gen=1, omega=0.7298, eta1=2.05, eta2=2.05, max_vel=0.5, variant=5, neighb_type=2,
        #                      neighb_param=4, memory=False, seed=random)

        self.population_size = 100
        self.verbosity = 1
        self.output_file = '1d_pygmo.csv'
