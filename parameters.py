import numpy as np


class ParametersSixD:
    """
    Class used to hold parameters needed to initialise CrossingSixD located in blocks.py

    Attributes
    ----------
    inv_c_charge : float
        The inverse central charge of the CFT. It can be set to 0 corresponding to the supergravity limit.
    spin_list_short_d : ndarray
        A NumPy array containing either [0] if the D[0,4] multiplet is present or [] if it isn't.
        No other values should be used.
    spin_list_short_b : ndarray
        A NumPy array containing a list of the B[0,2] multiplet spins. These must be even and given in increasing
        order without duplication.
    spin_list_long : ndarray
        A NumPy array containing a list of the L[0,0] long multiplet spins. These must be even and given in increasing
        order. Degeneracy of spins is allowed.
    ell_max : int
        Spin cutoff for the a_chi function in blocks.py.

    Notes
    -----
    No validation of the inputs is done.
    """

    def __init__(self):
        # ---Central charge---
        self.inv_c_charge = 1 / 25  # This is the inverse central charge (we can set it to 0 nicely i.e. infinite c)

        # ---Spin partition---
        # Note: spins HAVE to be given in ascending order
        self.spin_list_short_d = np.array([0])  # This can only be [] or [0]
        self.spin_list_short_b = np.array([2, 4, 6, 8, 10, 12, 14, 16])  # This cannot be empty
        self.spin_list_long = np.array([0, 0, 0, 0, 0, 0, 0, 0,
                                        2, 2, 2, 2, 2, 2, 2,
                                        4, 4, 4, 4, 4, 4,
                                        6, 6, 6, 6, 6,
                                        8, 8, 8, 8,
                                        10, 10, 10,
                                        12, 12,
                                        14])  # This cannot be empty but can have repeated entries
        # ---This is needed for the inhomogeneous part of the 6d crossing equation---
        self.ell_max = 30  # Spin cutoff for the a_chi function in blocks.py. This MUST be an even number

        # ---Pre-generated conformal block lattice parameters---
        self.delta_start = np.array([5.8, 7.8, 9.8, 11.8, 13.8, 15.8, 17.8, 19.8])
        self.delta_end_increment = 30.0
        self.delta_sep = 0.0005

        # This is a list of the original 180 columns to delete from the '6d_blocks_spin*.csv' files
        self.z_kill_list = []
        # An example of a non-empty z_kill_list
        # self.z_kill_list = [3, 5, 7, 9, 10, 11, 12, 13, 15, 18, 19, 22, 26,
        #                     27, 28, 29, 30, 31, 32, 35, 37, 40, 43, 44, 46, 48,
        #                     49, 53, 57, 58, 60, 61, 63, 65, 68, 70, 73, 74, 75,
        #                     79, 80, 81, 82, 83, 86, 87, 91, 95, 96, 102, 104, 107,
        #                     113, 114, 115, 116, 119, 121, 122, 124, 129, 131, 132, 134, 136,
        #                     139, 141, 144, 145, 146, 147, 149, 151, 153, 154, 155, 156, 157,
        #                     158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
        #                     171, 172, 173, 174, 175, 176, 177, 178, 179]

        # DO NOT CHANGE ANYTHING BEYOND THIS POINT IN THIS CLASS
        # ---Non-User Adjustable Parameters---
        self.num_of_operators_short_d = self.spin_list_short_d.size  # counts the number of D multiplet spins
        self.num_of_operators_short_b = self.spin_list_short_b.size  # counts the number of B multiplet spins
        self.num_of_operators_long = self.spin_list_long.size  # counts the number of long multiplet spins
        self.spin_list_short = np.concatenate((self.spin_list_short_d, self.spin_list_short_b))
        self.spin_list = np.concatenate((self.spin_list_short_d, self.spin_list_short_b,
                                         self.spin_list_long))
        self.multiplet_index = [np.arange(self.num_of_operators_short_d),
                                np.arange(self.num_of_operators_short_d,
                                          (self.num_of_operators_short_b + self.num_of_operators_short_d)),
                                np.arange((self.num_of_operators_short_b + self.num_of_operators_short_d),
                                          (self.num_of_operators_short_b + self.num_of_operators_short_d
                                           + self.num_of_operators_long))]


class ParametersSixD_SAC(ParametersSixD):
    """
    A subclass of ParametersSixD. It holds the parameters required to configure the soft-Actor-Critic algorithm.

    Attributes
    ----------
    filename_stem: str
        Stem of the filename where output is to be saved (.csv is appended automatically in code).
        Can be used to distinguish output from runs with different central charges e.g. sac_c25 or sac_cSUGRA.
    verbose : {'', 'e', 'o'}
        When the soft-Actor-Critic algorithm should print to the console:
        - ``: no output.
        - `e`: print everytime reward is recalculated.
        - `o`: only when faff_max is reached and a reinitialisation occurs.
    faff_max : int
        Maximum number of steps spent searching for an improved reward without success.
        When faff_max is reached a reinitialisation occurs.
        Higher value means algorithm spends more time searching for a better solution.
    pc_max : int
        Maximum number of reinitialisations before window size decrease.
        Higher value means algorithm spends more time searching for a better solution.
    window_rate : float
        Search window size decrease rate. Range is (0, 1).
        The window_rate multiplies the search window sizes so small values focus the search quickly.
    max_window_exp : int
        Maximum number of search window size decreases.
        The final window sizes will be equal to ( window_rate ** max_window_exp ) * guess_sizes.
    same_spin_hierarchy : bool
        This flag determines whether a minimum separation in scaling dimension of long operators of the same spin
        is enforced.
    dyn_shift : float
         The minimum separation in scaling dimension between long operators degenerate in spin.
    guessing_run_list_deltas : ndarray
        Controls the guessing mode status for each conformal weight datum.
        0 = non-guessing mode
        1 = guessing mode.
    guessing_run_list_opes : ndarray
        Controls the guessing mode status for each OPE-squared coefficient datum.
        0 = non-guessing mode
        1 = guessing mode
    guess_sizes_deltas : ndarray
        Initial size of search windows for the conformal weights. They need not all be the same value.
        The guess_sizes of short D and B multiplets should be set to 0 as their weights are fixed.
        There is an implicit upper bound set by the highest weight in the pregenerated conformal block csv files.
        They need not all be the same value.
    guess_sizes_opes : ndarray
        Initial size of search windows for the OPE-squared coefficients. They need not all be the same value.
    shifts_deltas : ndarray
        Lower bounds for the conformal weights. They need not all be the same value.
    shifts_opecoeffs : ndarray
        Lower bounds for the OPE-squared coefficients. They need not all be the same value.
    global_best : ndarray
        The CFT data to start the soft-Actor-Critic with.
        For a 'from scratch' run the values should be the same as guess_sizes_deltas and guess_sizes_opes.
    global_reward_start : float
        The initial reward to start with.
    action_space_N : ndarray
        The dimension of the search space, equal to twice the total number of operators.
    shifts : ndarray
        The concatenation of shifts_deltas and shifts_opes.
    guessing_run_list : ndarray
        The concatenation of guessing_run_list_deltas and guessing_run_list_opes.
    guess_sizes : ndarray
        The concatenation of guess_sizes_deltas and guess_sizes_opes.
    Notes
    -----
    The user should not modify the attributes action_space_N, shifts, guessing_run_list and guess_sizes.
    This subclass inherits the spin partition which must be defined in the class ParametersSixD.
    No validation of the inputs is done.
    """

    def __init__(self):
        super().__init__()

        # ---Output Parameters---
        self.filename_stem = 'sac_v2_'
        self.verbose = 'o'  # When the SAC algorithm should print to the console:
        # e - print at every step
        # o - only after a reinitialisation
        # default is '' which produces no output

        # ---Learn Loop Paramaters---
        self.faff_max = 50  # maximum time spent not improving

        # ---Automation Run Parameters---
        self.pc_max = 5  # max number of reinitialisations before window decrease
        self.window_rate = 0.7  # window decrease rate (between 1 and 0)
        self.max_window_exp = 8  # maximum number of window changes

        # ---Spin Hierachy Parameters---
        self.same_spin_hierarchy = True  # same long multiplet operators with the same spin should be ordered
        self.dyn_shift = 1.0  # set the gap between long multiplet same spin deltas

        # ---Environment Parameters---
        # set guessing run list for conformal weights
        self.guessing_run_list_deltas = np.array([0,
                                                  0, 0, 0, 0, 0, 0, 0, 0,
                                                  1, 1, 1, 1, 1, 1, 1, 1,
                                                  1, 1, 1, 1, 1, 1, 1,
                                                  1, 1, 1, 1, 1, 1,
                                                  1, 1, 1, 1, 1,
                                                  1, 1, 1, 1,
                                                  1, 1, 1,
                                                  1, 1,
                                                  1], dtype=bool)
        # set guessing run list for ope coefficients
        self.guessing_run_list_opes = np.array([1,
                                                1, 1, 1, 1, 1, 1, 1, 1,
                                                1, 1, 1, 1, 1, 1, 1, 1,
                                                1, 1, 1, 1, 1, 1, 1,
                                                1, 1, 1, 1, 1, 1,
                                                1, 1, 1, 1, 1,
                                                1, 1, 1, 1,
                                                1, 1, 1,
                                                1, 1,
                                                1], dtype=bool)
        # initial search window size for conformal weights
        # windows for D and B multiplets should be set to zero as they are fixed
        self.guess_sizes_deltas = np.array([0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                            10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                                            10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                                            10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                                            10.0, 10.0, 10.0, 10.0, 10.0,
                                            10.0, 10.0, 10.0, 10.0,
                                            10.0, 10.0, 10.0,
                                            10.0, 10.0,
                                            10.0])
        # initial search window size for OPE coeffs
        self.guess_sizes_opes = np.array([20,
                                          20, 20, 20, 20, 20, 20, 20, 20,
                                          20, 20, 20, 20, 20, 20, 20, 20,
                                          20, 20, 20, 20, 20, 20, 20,
                                          20, 20, 20, 20, 20, 20,
                                          20, 20, 20, 20, 20,
                                          20, 20, 20, 20,
                                          20, 20, 20,
                                          20, 20,
                                          20])
        # set minimum values for conformal weights
        # minimums for D and B multiplets are fixed as weights are known
        self.shifts_deltas = np.array([8.0,
                                       9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0,
                                       6.1, 6.1, 6.1, 6.1, 6.1, 6.1, 6.1, 6.1,
                                       8.1, 8.1, 8.1, 8.1, 8.1, 8.1, 8.1,
                                       10.1, 10.1, 10.1, 10.1, 10.1, 10.1,
                                       12.1, 12.1, 12.1, 12.1, 12.1,
                                       14.1, 14.1, 14.1, 14.1,
                                       16.1, 16.1, 16.1,
                                       18.1, 18.1,
                                       20.1])
        # set minimum values for OPE coeffs
        self.shifts_opecoeffs = np.array([0,
                                          0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0,
                                          0, 0, 0, 0,
                                          0, 0, 0,
                                          0, 0,
                                          0])

        # ---Starting Point Parameters---
        # initial configuration to explore around
        # set equal to combination of shifts_deltas and shifts_opecoeffs to effectively start from a zero solution
        self.global_best = np.array([8.0,
                                     9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0,
                                     6.1, 6.1, 6.1, 6.1, 6.1, 6.1, 6.1, 6.1,
                                     8.1, 8.1, 8.1, 8.1, 8.1, 8.1, 8.1,
                                     10.1, 10.1, 10.1, 10.1, 10.1, 10.1,
                                     12.1, 12.1, 12.1, 12.1, 12.1,
                                     14.1, 14.1, 14.1, 14.1,
                                     16.1, 16.1, 16.1,
                                     18.1, 18.1,
                                     20.1,
                                     0,
                                     0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0,
                                     0, 0, 0, 0,
                                     0, 0, 0,
                                     0, 0,
                                     0
                                     ])
        # initial reward to start with
        # set equal to 0.0 to start from a zero solution.
        self.global_reward_start = 0.0

        # ------------------------------------------------------
        # DO NOT CHANGE ANYTHING BEYOND THIS POINT IN THIS CLASS
        # ------------------------------------------------------
        # ---Non-User Adjustable Parameters---
        self.action_space_N = 2 * (self.num_of_operators_short_d + self.num_of_operators_short_b) \
                              + 2 * self.num_of_operators_long
        self.shifts = np.concatenate((self.shifts_deltas, self.shifts_opecoeffs))
        self.guessing_run_list = np.concatenate((self.guessing_run_list_deltas,
                                                 self.guessing_run_list_opes))
        self.guess_sizes = np.concatenate((self.guess_sizes_deltas, self.guess_sizes_opes))
        self.output_order = ''  # The order to arrange the data saved to csv file
        # default order is (index_array, reward, cft data)
        # 'Reversed' - order is (cft data, reward, index_array)
