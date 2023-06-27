import numpy as np


class ParametersTwoD_cft:
    """Class used to hold parameters needed to initialise CrossingTwoD located in environment.cfts.py

    Attributes
    __________
    hh : float
        The value of the external operator conformal weight divided by two.
    spin_list : ndarray
        A 1-D array of integer operator spins.
        This determines the number of operators in the truncated spectrum.
        s-channel operator spins must come first followed by t-channel spins.
        Within each channel the spins should be strictly increasing.
    block_type : list
        A list specifying the type of each operator in the truncated spectrum.
        The values to be used in the list are
            - 's1': s-channel, spin = 0,
            - 's2': s-channel, spin > 0,
            - 't1': t-channel, spin = 0,
            - 't2': t-channel, spin > 0.
    same_spin_hierarchy : bool
        A value of True imposes a minimum separation in conformal weights for spin degenerate operators.
        Should be set to False for pygmo runs but can be True or False for SAC runs.
    dyn_shifts : ndarray
        A 1-D array of values for the minimum separation of adjacent spin degenerate operators.
        Only used if `same_spin_hierarchy` is True.
    delta_start : ndarray
        A 1-D array of values, for each unique spin, of the lowest conformal weight
        used to pre-generate the conformal block data.
    delta_sep : float
        The increase in conformal weights between adjacent
        lattice points in the pre-generated conformal block data.
    delta_end : float
        The value of the highest conformal weight above `delta_start`
        used to pre-generate the conformal block data.
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
        Practically, for an operator of spin s, no value should be larger than
        `delta_start[s]` + `delta_end` to ensure staying within the range accessible
        in the pre-generated conformal block data.
    lb_opecoeffs : ndarray
        A 1-D array holding strict lower bounds for the OPE coefficients.
    ub_opecoeffs : ndarray
        A 1-D array holding strict upper bounds for the OPE coefficients.
    include_tail : bool
        Has no effect as the analytical tail contribution is not yet implemented.

    """

    def __init__(self):
        # ---external operator weight/2 i.e. cft coupling---
        self.hh = 0.05

        # ---Spin partition---
        self.spin_list = np.array([0, 0, 1, 2, 3, 4, 5,
                                   0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 5])  # spin partition, ensure correct order!
        self.block_type = ['s1', 's1', 's2', 's2', 's2', 's2', 's2',
                           't1', 't1', 't2', 't2', 't2', 't2', 't2', 't2', 't2', 't2', 't2']
        # type s1: g_a (scalar, s channel),
        # type s2: g_a_symm (spin > 0, s channel),
        # type t1: g_b (scalar, t channel),
        # type t2: g_b_symm (spin > 0, t channel)

        # ---Spin Hierarchy Parameters---
        self.same_spin_hierarchy = True  # same long multiplet operators with the same spin should be ordered
        self.dyn_shifts = np.array([3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                    1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 0.0, 0.0])

        # ---Pre-generated conformal block lattice Parameters---
        self.delta_start = np.array([-0.2, 0.8, 1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8, 8.8])
        self.delta_sep = 0.00092
        self.delta_end = 9.0
        self.z_kill_list = []

        # ---Bounds for CFT data---
        self.lb_deltas = np.array([0.1, 0.1, 0.8, 1.8, 2.8, 3.8, 4.8,
                                   1.8, 1.8, 0.8, 0.8, 0.8, 1.8, 1.8, 2.8, 2.8, 3.8, 4.8])
        self.lb_opecoeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.ub_deltas = 10 + np.array([0.0, 0.0, 0.8, 1.8, 2.8, 3.8, 4.8,
                                        1.8, 1.8, 0.8, 0.8, 0.8, 1.8, 1.8, 2.8, 2.8, 3.8, 4.8])
        self.ub_opecoeffs = 10 + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # ---Analytic tail---
        self.include_tail = True  # has no effect as not yet implemented in CrossingTwoD.


class ParametersTwoD_sac:
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
        self.filename_stem = '2d_sac'
        self.verbose = 'e'
        self.output_order = ''
        self.faff_max = 300
        self.pc_max = 5
        self.window_rate = 0.7
        self.max_window_exp = 10
        self.guessing_run_list_deltas = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                  0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
        self.guessing_run_list_opes = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
        self.guess_sizes_deltas = np.array([1.0, 5.0, 6.0, 3.0, 4.0, 5.0, 6.0, 3.0, 5.0, 2.0, 4.0,
                                            6.0, 3.0, 5.0, 4.0, 6.0, 5.0, 6.0])
        self.guess_sizes_opes = np.array([2.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                                          0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        self.global_reward_start = 0.0
        self.global_reward_start = 0.0
        self.global_best = np.array([0.059848226606845856, 3.059848226606846, 0.8, 1.8, 2.8, 3.8, 4.8, 1.8, 3.3, 0.8,
                                     2.3, 3.8, 1.8, 3.3, 2.8, 4.3, 3.8, 4.8, 0.8877384857632966, 0.08314371633294754,
                                     0.025873235346901058, 0.0, 0.0, 0.006355320001588264, 0.0, 0.0008458751201553063,
                                     0.13949767101881141, -0.059062997983763785, -0.0003400318631835692,
                                     -0.032149169427732126, 0.09453362226486206, 0.2, 0.0, -0.2, 0.0,
                                     -0.009933355127662694])


class Parameters_pygmo:
    """Class used to hold parameters needed to initialise and customise a pygmo run."""

    def __init__(self):
        """
        udp : {'udp_basic'}
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
        self.output_file = '2d_pygmo.csv'
