import numpy as np


class ParametersSixD_cft:
    """
    Class used to hold parameters needed to initialise CrossingSixD located in environment.cfts.py

    Attributes
    ----------
    inv_c_charge : float
        The inverse central charge of the CFT. It can be set to 0 corresponding to the supergravity limit.
    spin_list_short_d : ndarray
        A 1-D array containing [0] for the D[0,4] multiplet.
        No other value should be used.
    spin_list_short_b : ndarray
        A 1-D array containing a list of the B[0,2] multiplet spins appearing in the truncated spectrum.
        These must be even and given in strictly increasing order without duplication.
    spin_list_long : ndarray
        A 1-D array containing a list of the L[0,0] long multiplet spins appearing in the truncated spectrum.
        These must be even and given in increasing order. Degeneracy of spins is allowed.
    ell_max : int
        Spin cutoff for the a_chi function in CrossingSixD. This must be an even number.
    same_spin_hierarchy : bool
        A value of True imposes a minimum separation in conformal weights for spin degenerate long operators.
        Should be set to False for pygmo runs but can be True or False for SAC runs.
    dyn_shifts : ndarray
        A 1-D array of values for the minimum separation of adjacent spin degenerate long operators.
        Only used if `same_spin_hierarchy` is True.
    delta_start : ndarray
        A 1-D array of values, for each unique spin, of the lowest conformal weight
        used to pre-generate the conformal block data.
    delta_sep : float
        The increase in conformal weights between adjacent
        lattice points in the pre-generated conformal block data.
    delta_end_increment : float
        The value of the highest conformal weight above `delta_start`
        used to pre-generate the conformal block data.
        Controls which pre-generated blocks are loaded.
    z_kill_list : list
        A list of positions to delete from the pre-generated conformal block data arrays.
        If a pre-generated conformal block data array has shape (M,N) then
        specifying a non-empty `z_kill_list` will lead to an array of shape
        (M, N - len(z_kill_list)).
    lb_deltas : ndarray
        A 1-D array holding strict lower bounds for the conformal weights of long operators only
        (D and B operators have fixed conformal weights so should not be included).
    ub_deltas : ndarray
        A 1-D array holding strict upper bounds for the conformal weights of long operators only
        (D and B operators have fixed conformal weights so should not be included).
        Practically, for long operators of spin s, no value should be larger than
        `delta_start[s]` + `delta_end` to ensure staying within the range accessible
        in the pre-generated conformal block data.
    lb_opecoeffs : ndarray
        A 1-D array holding strict lower bounds for the OPE coefficients.
        The D[0,4] multiplet is first followed by the B[0,2] multiplets and finally the long multiplets.
    ub_opecoeffs : ndarray
        A 1-D array holding strict upper bounds for the OPE coefficients.
        The D[0,4] multiplet is first followed by the B[0,2] multiplets and finally the long multiplets.
    include_tail : bool
        Has no effect as the analytical tail contribution is not yet implemented.

    Notes
    -----
    No validation of the inputs is done.
    """

    def __init__(self):
        # ---Central charge i.e. cft coupling---
        self.inv_c_charge = 0.0  # This is the inverse central charge (we can set it to 0 nicely i.e. infinite c)

        # ---Spin partition---
        # Note: spins HAVE to be given in ascending order
        self.spin_list_short_d = np.array([0])  # This can only be [0]
        self.spin_list_short_b = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18])  # This cannot be empty
        self.spin_list_long = np.array([0, 0, 0, 0, 0, 0, 0, 0,
                                        2, 2, 2, 2, 2, 2, 2,
                                        4, 4, 4, 4, 4, 4,
                                        6, 6, 6, 6, 6,
                                        8, 8, 8, 8,
                                        10, 10, 10,
                                        12, 12])  # This cannot be empty but can have repeated entries

        self.ell_max = 30  # Spin cutoff for the a_chi function in environment.cfts.py. This MUST be an even number.

        # ---Spin Hierachy Parameters---
        self.same_spin_hierarchy = True  # same long multiplet operators with the same spin should be ordered
        self.dyn_shifts = 0.3 * np.ones(35)  # set the gap between long multiplet same spin deltas

        # ---Pre-generated conformal block lattice parameters---
        self.delta_start = np.array([5.8, 7.8, 9.8, 11.8, 13.8, 15.8, 17.8, 19.8])
        self.delta_sep = 0.0005
        self.delta_end_increment = 30.0
        self.z_kill_list = []

        # ---Bounds for CFT data---
        # set minimum values for conformal weights of the long operators
        self.lb_deltas = np.array([6.1, 6.1, 6.1, 6.1, 6.1, 6.1, 6.1, 6.1,
                                   8.1, 8.1, 8.1, 8.1, 8.1, 8.1, 8.1,
                                   10.1, 10.1, 10.1, 10.1, 10.1, 10.1,
                                   12.1, 12.1, 12.1, 12.1, 12.1,
                                   14.1, 14.1, 14.1, 14.1,
                                   16.1, 16.1, 16.1,
                                   18.1, 18.1])
        # set minimum values for OPE coeffs of D,B and long
        self.lb_opecoeffs = np.array([0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0,
                                      0, 0, 0, 0,
                                      0, 0, 0,
                                      0, 0])
        # set maximum values for conformal weights of the long operators
        self.ub_deltas = np.array([6.1 + 29.7, 6.1 + 29.7, 6.1 + 29.7, 6.1 + 29.7, 6.1 + 29.7, 6.1 + 29.7, 6.1 + 29.7,
                                   6.1 + 29.7,
                                   8.1 + 29.7, 8.1 + 29.7, 8.1 + 29.7, 8.1 + 29.7, 8.1 + 29.7, 8.1 + 29.7, 8.1 + 29.7,
                                   10.1 + 29.7, 10.1 + 29.7, 10.1 + 29.7, 10.1 + 29.7, 10.1 + 29.7, 10.1 + 29.7,
                                   12.1 + 29.7, 12.1 + 29.7, 12.1 + 29.7, 12.1 + 29.7, 12.1 + 29.7,
                                   14.1 + 29.7, 14.1 + 29.7, 14.1 + 29.7, 14.1 + 29.7,
                                   16.1 + 29.7, 16.1 + 29.7, 16.1 + 29.7,
                                   18.1 + 29.7, 18.1 + 29.7
                                   ])
        # set maximum values for OPE coeffs of D,B and long
        self.ub_opecoeffs = 100 * np.array([1,
                                            1, 1, 1, 1, 1, 1, 1, 1, 1,
                                            1, 1, 1, 1, 1, 1, 1, 1,
                                            1, 1, 1, 1, 1, 1, 1,
                                            1, 1, 1, 1, 1, 1,
                                            1, 1, 1, 1, 1,
                                            1, 1, 1, 1,
                                            1, 1, 1,
                                            1, 1])

        # ---Analytic tail---
        self.include_tail = True  # has no effect as not yet implemented in CrossingSixD.


class ParametersSixD_sac:
    """
    It holds the parameters required to configure the soft-Actor-Critic algorithm.

    Attributes
    ----------
    filename_stem: str
        Stem of the filename where output is to be saved (.csv is appended automatically in code).
        Can be used to distinguish output from runs with different central charges e.coupling. sac_c25 or sac_cSUGRA.
    verbose : {'', 'e', 'o'}
        When the soft-Actor-Critic algorithm should print to the console:
        - ``: no output.
        - `e`: print everytime reward is recalculated.
        - `o`: only when faff_max is reached and a re-initialisation occurs.
    output_order : {'', 'Reversed'}
        The order to arrange the data saved to csv file
        - ``: default order is (index_array, reward, cft data)
        - `Reversed` - order is (cft data, reward, index_array)
    faff_max : int
        Maximum number of steps spent searching for an improved reward without success.
        When faff_max is reached a re-initialisation occurs.
        Higher value means algorithm spends more time searching for a better solution.
    pc_max : int
        Maximum number of re-initialisations before window size decrease.
        Higher value means algorithm spends more time searching for a better solution.
    window_rate : float
        Search window size decrease rate. Range is (0, 1).
        The window_rate multiplies the search window sizes so small values focus the search quickly.
    max_window_exp : int
        Maximum number of search window size decreases.
        The final window sizes will be equal to ( window_rate ** max_window_exp ) * guess_sizes.
    guessing_run_list_deltas : ndarray
        Controls the guessing mode status for each long operator conformal weight
        (D and B operators have fixed conformal weights so should not be included).
        0 = non-guessing mode
        1 = guessing mode.
    guessing_run_list_opes : ndarray
        Controls the guessing mode status for each OPE-squared coefficient datum (D, B and long multiplets).
        0 = non-guessing mode
        1 = guessing mode
    guess_sizes_deltas : ndarray
        Initial size of search windows for the conformal weights of the long operators
        (D and B operators have fixed conformal weights so should not be included).
        They need not all be the same value.
        The guess_sizes of short D and B multiplets should be set to 0 as their weights are fixed.
        There is an implicit upper bound set by the highest weight in the pregenerated conformal block csv files.
        They need not all be the same value.
    guess_sizes_opes : ndarray
        Initial size of search windows for the OPE-squared coefficients. They need not all be the same value.
    global_reward_start : float
        The initial reward to start with.
    global_best : ndarray
        The CFT data to start the soft-Actor-Critic with.
        For a 'from scratch' run the values should be the same as guess_sizes_deltas and guess_sizes_opes.

    Notes
    -----
    No validation of the inputs is done.
    """

    def __init__(self):
        # ---Output Parameters---
        self.filename_stem = '6d_sac'
        self.verbose = 'e'
        self.output_order = ''
        self.faff_max = 300
        self.pc_max = 5
        self.window_rate = 0.7
        self.max_window_exp = 10
        self.guessing_run_list_deltas = np.array([1, 1, 1, 1, 1, 1, 1, 1,
                                                  1, 1, 1, 1, 1, 1, 1,
                                                  1, 1, 1, 1, 1, 1,
                                                  1, 1, 1, 1, 1,
                                                  1, 1, 1, 1,
                                                  1, 1, 1,
                                                  1, 1], dtype=bool)
        self.guessing_run_list_opes = np.array([1,
                                                1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                1, 1, 1, 1, 1, 1, 1, 1,
                                                1, 1, 1, 1, 1, 1, 1,
                                                1, 1, 1, 1, 1, 1,
                                                1, 1, 1, 1, 1,
                                                1, 1, 1, 1,
                                                1, 1, 1,
                                                1, 1], dtype=bool)
        self.guess_sizes_deltas = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                                            10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                                            10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                                            10.0, 10.0, 10.0, 10.0, 10.0,
                                            10.0, 10.0, 10.0, 10.0,
                                            10.0, 10.0, 10.0,
                                            10.0, 10.0])
        self.guess_sizes_opes = np.array([20,
                                          20, 20, 20, 20, 20, 20, 20, 20, 20,
                                          20, 20, 20, 20, 20, 20, 20, 20,
                                          20, 20, 20, 20, 20, 20, 20,
                                          20, 20, 20, 20, 20, 20,
                                          20, 20, 20, 20, 20,
                                          20, 20, 20, 20,
                                          20, 20, 20,
                                          20, 20])
        self.global_reward_start = 0.0
        self.global_best = np.array([6.1, 6.1, 6.1, 6.1, 6.1, 6.1, 6.1, 6.1,
                                     8.1, 8.1, 8.1, 8.1, 8.1, 8.1, 8.1,
                                     10.1, 10.1, 10.1, 10.1, 10.1, 10.1,
                                     12.1, 12.1, 12.1, 12.1, 12.1,
                                     14.1, 14.1, 14.1, 14.1,
                                     16.1, 16.1, 16.1,
                                     18.1, 18.1,
                                     0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0,
                                     0, 0, 0, 0,
                                     0, 0, 0,
                                     0, 0])


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
        self.udp = 'udp_basic'  # {'udp_basic', 'udp_1d', 'udp_1d_integral_constraints'}
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
        self.output_file = '6d_pygmo.csv'
