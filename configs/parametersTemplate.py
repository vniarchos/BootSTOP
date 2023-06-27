import numpy as np


class Parameters_sac:
    """
    Class used to hold the parameters required to configure the soft-Actor-Critic (SAC) algorithm.

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
    Notes
    -----
    No validation of the inputs is done.
    """

    def __init__(self):
        self.filename_stem = 'sac'
        self.verbose = 'o'  # {'', 'e', 'o'}
        self.output_order = ''  # {'', 'Reversed'}
        self.faff_max = 300
        self.pc_max = 5
        self.window_rate = 0.7
        self.max_window_exp = 10
        self.guessing_run_list_deltas = np.array([], dtype=bool)
        self.guessing_run_list_opes = np.array([], dtype=bool)
        self.guess_sizes_deltas = np.array([])
        self.guess_sizes_opes = np.array([])
        self.global_reward_start = 0.0
        self.global_best = np.array([])


class Parameters_pygmo:
    """
    Class used to hold parameters needed to initialise and customise a pygmo run.

    Attributes
    ----------
    udp : str
        String holding the name of the user defined problem.
        It must match the .__name__ of one of the classes in environment.pygmo_udps.py
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

    def __init__(self):
        self.udp = ''
        self.pygmo_algo_name = ''
        self.pygmo_algo_dict = dict()
        self.population_size = 1
        self.verbosity = 1
        self.output_file = 'pygmo_output.csv'
