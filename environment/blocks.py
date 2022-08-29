import numpy as np
import mpmath as mp
import scipy.special as sc
from numpy import linalg as LA


class CrossingSixD:
    """
    Computes truncated crossing equation for 6d (2,0) SCFT as in 1507.05637.

    Parameters
    ----------
    block_list : array_like
        A NumPy array containing the pregenerated conformal block data.
    params : ParametersSixD
        Instance of `ParametersSixD` class.
    z_data : ZData
        Instance of `ZData` class.
    """

    def __init__(self, block_list, params, z_data):
        self.spin_list_short_b = params.spin_list_short_b
        self.spin_list_short = params.spin_list_short
        self.spin_list_long = params.spin_list_long
        self.inv_c_charge = params.inv_c_charge
        self.action_space_N = params.action_space_N
        self.ell_max = params.ell_max
        self.multiplet_index = params.multiplet_index
        self.num_of_short = params.num_of_operators_short_d + params.num_of_operators_short_b
        self.num_of_long = params.num_of_operators_long
        self.delta_sep = params.delta_sep
        self.delta_start = params.delta_start
        self.delta_end_increment = params.delta_end_increment

        self.env_shape = z_data.env_shape
        self.z = z_data.z
        self.z_conj = z_data.z_conj

        self.use_scipy_for_hypers = True  # compute the hypergeometrics using scipy package or mpmath
        self.block_list = block_list

        self.INHO_VALUE = self.inhomo_z_vector()
        self.SHORT_D_HYPERS = self.short_coeffs_d_multiplet()
        self.SHORT_B_HYPERS = self.short_coeffs_b_multiplet_array()

    def h(self, z):
        """Computes h(z) defined in eqn (3.21) of 1507.05637."""
        res_a = - ((1 / 3) * (z ** 3) - (z - 1) ** (-1) - (z - 1) ** (-2) - (1 / 3) * (z - 1) ** (-3) - z ** (-1)) \
                - 8 * self.inv_c_charge * (z - (z - 1) ** (-1) + np.log(1 - z)) - (1 / 6) + 8 * self.inv_c_charge
        return res_a

    def c_h(self, z, zb):
        """Computes C(z,zb) defined in eqn (2.15) of 2105.13361."""
        res_a = (z - zb) ** (-3) * (z * zb) ** (-1) * (self.h(z) - self.h(zb))
        return res_a

    def f_nm(self, n, m, ell, delta, z, zb, delta12, delta34):
        """Computes the function defined after (B.1) of 1507.05637."""
        prefactor = (z * zb) ** ((delta - ell) / 2) * (z - zb) ** (-3)
        if self.use_scipy_for_hypers:
            res_zzb = prefactor * (
                    ((- z / 2) ** ell) * (z ** (n + 3)) * (zb ** m) *
                    sc.hyp2f1(((delta + ell - delta12) / 2) + n,
                              ((delta + ell + delta34) / 2) + n, delta + ell + 2 * n, z) *
                    sc.hyp2f1(((delta - ell - delta12) / 2) - 3 + m, ((delta - ell + delta34) / 2) - 3 + m,
                              delta - ell - 6 + 2 * m, zb))

            res_zbz = prefactor * (
                    ((- zb / 2) ** ell) * (zb ** (n + 3)) * (z ** m) *
                    sc.hyp2f1(((delta + ell - delta12) / 2) + n,
                              ((delta + ell + delta34) / 2) + n, delta + ell + 2 * n, zb) *
                    sc.hyp2f1(((delta - ell - delta12) / 2) - 3 + m, ((delta - ell + delta34) / 2) - 3 + m,
                              delta - ell - 6 + 2 * m, z))
        else:
            res_zzb = prefactor * (
                    ((- z / 2) ** ell) * (z ** (n + 3)) * (zb ** m) *
                    mp.hyp2f1(((delta + ell - delta12) / 2) + n,
                              ((delta + ell + delta34) / 2) + n, delta + ell + 2 * n, z) *
                    mp.hyp2f1(((delta - ell - delta12) / 2) - 3 + m, ((delta - ell + delta34) / 2) - 3 + m,
                              delta - ell - 6 + 2 * m, zb))

            res_zbz = prefactor * (
                    ((- zb / 2) ** ell) * (zb ** (n + 3)) * (z ** m) *
                    mp.hyp2f1(((delta + ell - delta12) / 2) + n,
                              ((delta + ell + delta34) / 2) + n, delta + ell + 2 * n, zb) *
                    mp.hyp2f1(((delta - ell - delta12) / 2) - 3 + m, ((delta - ell + delta34) / 2) - 3 + m,
                              delta - ell - 6 + 2 * m, z))

        res = res_zzb - res_zbz
        return res

    def g_l_delta(self, ell, delta, z, zb, delta12=0, delta34=-2):
        """Computes the function defined in (B.1) of 1507.05637."""
        numerator3 = (delta - 4) * (ell + 3) * (delta - ell - delta12 - 4) * (delta - ell + delta12 - 4) \
                     * (delta - ell + delta34 - 4) * (delta - ell - delta34 - 4)
        denominator3 = 16 * (delta - 2) * (ell + 1) * (delta - ell - 5) * (delta - ell - 4) ** 2 * (delta - ell - 3)
        g_l_delta_coeff_3 = numerator3 / denominator3

        numerator4 = - (delta - 4) * (delta + ell - delta12) * (delta + ell + delta12) * (delta + ell + delta34) \
                     * (delta + ell - delta34)
        denominator4 = 16 * (delta - 2) * (delta + ell - 1) * (delta + ell) ** 2 * (delta + ell + 1)
        g_l_delta_coeff_4 = numerator4 / denominator4

        numerator5 = 2 * (delta - 4) * (ell + 3) * delta12 * delta34
        denominator5 = (delta + ell) * (delta + ell - 2) * (delta + ell - 4) * (delta + ell - 6)
        g_l_delta_coeff_5 = numerator5 / denominator5

        res_a = self.f_nm(0, 0, ell, delta, z, zb, delta12, delta34) \
                - (ell + 3) / (ell + 1) * self.f_nm(-1, 1, ell, delta, z, zb, delta12, delta34) \
                + g_l_delta_coeff_3 * self.f_nm(0, 2, ell, delta, z, zb, delta12, delta34) \
                + g_l_delta_coeff_4 * self.f_nm(1, 1, ell, delta, z, zb, delta12, delta34) \
                + g_l_delta_coeff_5 * self.f_nm(0, 1, ell, delta, z, zb, delta12, delta34)
        return res_a

    def b_l(self, ell):
        """Computes the b_l coefficients given in (4.9) of 1507.05637."""
        half_ell = ell // 2  # This is needed to avoid error in factorial2 as it doesn't like float variables

        numerator1 = (ell + 1) * (ell + 3) * (ell + 2) ** 2 * sc.factorial(half_ell) \
                     * sc.factorial2(half_ell + 2, exact=True) * sc.factorial2(half_ell + 3, exact=True) \
                     * sc.factorial2(ell + 5, exact=True)
        numerator2 = self.inv_c_charge * 8 * (2 ** (-(half_ell + 1))) * (ell * (ell + 7) + 11) \
                     * sc.factorial2(ell + 3, exact=True) * sc.gamma(half_ell + 2)
        denominator1 = 18 * sc.factorial2(ell + 2, exact=True) * sc.factorial2(2 * ell + 5, exact=True)
        denominator2 = sc.factorial2(2 * ell + 5, exact=True)
        res_a = numerator1 / denominator1 + numerator2 / denominator2
        return res_a

    def a_atomic(self, delta, ell, z, zb):
        """Computes the function 'a' in (4.5) of 1507.05637."""
        res_a = 4 * ((z ** 6) * (zb ** 6) * (delta - ell - 2) * (delta + ell + 2)) ** (-1) \
                * self.g_l_delta(ell, delta + 4, z, zb)
        return res_a

    def a_chi(self, z, zb):
        """Computes a truncated version of the function given in (4.11) of 1507.05637."""
        res_a = np.sum((2 ** k) * self.b_l(k)
                       * self.a_atomic(k + 4, k, z, zb) for k in range(0, self.ell_max + 2, 2))
        # sum runs over even spins to upto (and including) ell_max.
        return res_a

    def inhomo_z_vector(self):
        """
        Computes (a truncated version of) the RHS of (4.13) in 1507.05637 for all points in the z-sample
        simultaneously.
        """
        res_inhomo_z_vector = - ((self.z - self.z_conj) ** (-3)) * (
                (self.h(1 - self.z_conj) - self.h(1 - self.z)) * (((self.z - 1) * (self.z_conj - 1)) ** (-1))
                + (self.h(self.z_conj) - self.h(self.z)) * ((self.z * self.z_conj) ** (-1))) \
                              - (self.z - 1) * (self.z_conj - 1) * self.a_chi(1 - self.z, 1 - self.z_conj) \
                              + self.z * self.z_conj * self.a_chi(self.z, self.z_conj)
        return res_inhomo_z_vector.real

    def short_coeffs_b_multiplet(self, ell):
        """
        Computes the contribution to the crossing equation of a single spin ell B multiplet for all points in the
        z-sample simultaneously.

        Returns
        -------
        ndarray(env_shape,)

        Notes
        -----
        Since the conformal weight of a B[0,2] multiplet is fixed by the spin and only needs computing once,
        it's as quick to compute using scipy or mpmath rather than picking from the pregenerated data.
        """
        res = (self.z * self.z_conj * self.a_atomic(ell + 6, ell, self.z, self.z_conj)
               - (self.z - 1) * (self.z_conj - 1) * self.a_atomic(ell + 6, ell, 1 - self.z, 1 - self.z_conj)).real
        return res

    def short_coeffs_b_multiplet_array(self):
        """
        Aggregates the B multiplet contributions of differing spins together into an array.

        Returns
        -------
        ndarray(spin_list_short_b.size, env_shape)
        """
        # we assume there's always at least one B multiplet
        b_multiplet_array = self.short_coeffs_b_multiplet(self.spin_list_short_b[0])
        # if there's only a single B multiplet we don't need to stack
        if self.spin_list_short_b.size == 1:
            return b_multiplet_array
        # otherwise we vertically stack the contribution for each spin
        else:
            b_multiplet_spins = self.spin_list_short_b[1:]
            for ell in b_multiplet_spins:
                b_multiplet_array = np.vstack((b_multiplet_array, self.short_coeffs_b_multiplet(ell)))

        return b_multiplet_array

    def short_coeffs_d_multiplet(self):
        """
        Computes the contribution to the crossing equation of a single spin D multiplet for all points in the z-sample
        simultaneously.

        Returns
        -------
        ndarray(env_shape,)

        Notes
        -----
        Since the conformal weight of a D[0,4] multiplet is fixed by the spin and only needs computing once,
        it's as quick to compute using scipy or mpmath rather than picking from the pregenerated data.

        """
        return (self.z * self.z_conj * self.a_atomic(6, 0, self.z, self.z_conj)
                - (self.z - 1) * (self.z_conj - 1) * self.a_atomic(6, 0, 1 - self.z, 1 - self.z_conj)).real

    # def long_cons_non_pregenerated(self, delta, ell):
    #    long_c = (self.z * self.z_conj * self.a_atomic(delta, ell, self.z, self.z_conj)
    #              - (self.z - 1) * (self.z_conj - 1) * self.a_atomic(delta, ell, 1 - self.z, 1 - self.z_conj)).real
    #    return long_c

    def long_cons(self, delta, ell):
        """
        Computes the (pregenerated) contribution to the crossing equation of a single spin long multiplet for all
        points in the z-sample simultaneously.

        Parameters
        ----------
        delta : float
            The conformal weight of the long multiplet.
        ell : int
            The spin of the long multiplet.
        Returns
        -------
        ndarray(env_shape,)
        """
        ell = ell // 2  # make sure the spin is even
        # we clip delta to ensure that it lies within the range accessible in the pregenerated blocks
        delta = np.clip(delta, a_min=None, a_max=self.delta_start[ell] + self.delta_end_increment - self.delta_sep)
        # used a_min=None in delta because a lower bound is enforced by choosing shifts_deltas in the parameters.py file
        # now we find the nearest lattice point delta corresponds to
        n = np.rint((delta - self.delta_start[ell]) / self.delta_sep)
        # get the appropriate contribution from block_list based on spin and lattice point
        long_c = self.block_list[ell][int(n)]
        # we need to transpose to return a shape compatible with the short multiplet contributions
        return np.transpose(long_c)

    def long_coeffs_array(self, deltas):
        """
        Aggregates all the long multiplet contributions together into a single array.

        Returns
        -------
        long_c : ndarray(num_of_long, env_shape)
        """
        long_c = self.long_cons(deltas[0], self.spin_list_long[0])
        for x in range(1, deltas.size):
            long_c = np.vstack((long_c, self.long_cons(deltas[x], self.spin_list_long[x])))
        return long_c


class CrossingSixD_SAC(CrossingSixD):

    def __init__(self, block_list, params, z_data):
        super().__init__(block_list, params, z_data)

        self.same_spin_hierarchy_deltas = params.same_spin_hierarchy  # impose weight separation flag
        self.dyn_shift = params.dyn_shift  # the weight separation value
        self.dup_list = self.spin_list_long == np.roll(self.spin_list_long, -1)  # which long spins are degenerate

    def split_cft_data(self, cft_data):
        """
        Sets up dictionaries to decompose the search space data into easily identifiable pieces.

        Parameters
        ----------
        cft_data : ndarray
            The array to be split.

        Returns
        -------
        delta_dict : dict
            A dictionary containing the keys ("short_d", "short_b", "long") and values of the conformal weights.
        ope_dict : dict
            A dictionary containing the keys ("short_d", "short_b", "long") and values of the OPE-squared coefficients.

        """
        delta_dict = {
            "short_d": cft_data[self.multiplet_index[0]],
            "short_b": cft_data[self.multiplet_index[1]],
            "long": cft_data[self.multiplet_index[2]]
        }
        ope_dict = {
            "short_d": cft_data[self.multiplet_index[0] + self.action_space_N // 2],
            "short_b": cft_data[self.multiplet_index[1] + self.action_space_N // 2],
            "long": cft_data[self.multiplet_index[2] + self.action_space_N // 2]
        }
        return delta_dict, ope_dict

    def impose_weight_separation(self, delta_dict):
        """
        Enforces a minimum conformal dimension separation between long multiplets of the same spin by
        overwriting values of delta_dict.

        Parameters
        ----------
        delta_dict : dict
            A dictionary of multiplet types and their conformal weights.
        Returns
        -------
        delta_dict : dict
            Dictionary with modified values for 'long' key.
        """
        deltas = delta_dict['long']
        flag_current = False
        flag_next = False
        for i in range(self.dup_list.size):
            flag_current = self.dup_list[i]
            flag_next_tmp = False

            if flag_next and not flag_current:
                deltas[i] = np.clip(deltas[i], a_min=(deltas[i - 1] + self.dyn_shift), a_max=None)

            if flag_current and not flag_next:
                flag_next_tmp = True

            if flag_current and flag_next:
                deltas[i] = np.clip(deltas[i], a_min=(deltas[i - 1] + self.dyn_shift), a_max=None)
                flag_next_tmp = True

            flag_next = flag_next_tmp

        return delta_dict

    def crossing(self, cft_data):
        """
        Evaluates the truncated crossing equations for the given CFT data at all points in the z-sample simultaneously.

        Parameters
        ----------
        cft_data : ndarray
            An array containing the conformal weights and OPE-squared coefficients of all the multiplets.

        Returns
        -------
        constraints : ndarray
            Array of values of the truncated crossing equation.
        reward : float
            The reward determined from the constraints.
        cft_data : ndarray
            A list of possibly modified CFT data.

        """
        # get some dictionaries
        delta_dict, ope_dict = self.split_cft_data(cft_data)

        if self.same_spin_hierarchy_deltas:
            # impose the mimimum conformal weight separations between operators
            delta_dict = self.impose_weight_separation(delta_dict)
            # since we've altered some data we update the long multiplet weights in cft_data
            cft_data[self.multiplet_index[2]] = delta_dict['long']

        if len(delta_dict['short_d']) == 0:
            # broadcast a zero ope coefficient over the crossing contributions
            short_cons_d_multiplet = 0 * self.SHORT_D_HYPERS
        else:
            # broadcast the D multiplet ope coefficient over the crossing contributions
            short_cons_d_multiplet = ope_dict['short_d'] * self.SHORT_D_HYPERS

        # broadcast the reshaped B multiplet ope coefficients over their crossing contributions
        short_cons_b_multiplet = ope_dict['short_b'].reshape(-1, 1) * self.SHORT_B_HYPERS
        # broadcast the reshaped long multiplet ope coefficients over their crossing contributions
        long_cons = ope_dict['long'].reshape(-1, 1) * self.long_coeffs_array(delta_dict['long'])
        # long_cons.shape = (num_of_long, env_shape)

        # add up all the components
        constraints = self.INHO_VALUE + short_cons_d_multiplet + short_cons_b_multiplet.sum(axis=0) \
                      + long_cons.sum(axis=0)  # the .sum implements summation over multiplet spins
        reward = 1 / LA.norm(constraints)

        return constraints, reward, cft_data
