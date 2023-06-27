import numpy as np
import scipy.special as sc
import mpmath as mp
import environment.utils as utils


class CrossingSixD:
    """Computes truncated crossing equation for 6d (2,0) SCFT as in 1507.05637.

    """

    def __init__(self, params, z_data):
        """
        Parameters
        ----------
        params : ParametersSixD_cft
            Instance of `ParametersSixD_cft` class.
        z_data : ZData
            Instance of `ZData` class.

        """
        self.spin_list_short_d = params.spin_list_short_d
        self.spin_list_short_b = params.spin_list_short_b
        self.spin_list_short = np.concatenate((self.spin_list_short_d, self.spin_list_short_b))
        self.spin_list_long = params.spin_list_long
        self.coupling = params.inv_c_charge
        self.action_space_N = self.spin_list_short_d.size + self.spin_list_short_b.size \
                              + 2 * self.spin_list_long.size

        self.ell_max = params.ell_max
        self.num_of_short = self.spin_list_short_d.size + self.spin_list_short_b.size
        self.num_of_long = self.spin_list_long.size

        self.delta_sep = params.delta_sep
        self.delta_start = params.delta_start
        self.delta_end_increment = params.delta_end_increment

        self.env_dim = z_data.env_dim
        self.z = z_data.z
        self.z_conj = z_data.z_conj

        # ---Load the pre-generated conformal blocks for a_chi---
        blocks_chi = utils.generate_block_list_npy('6d/6d_blocks_chi', [''], params.z_kill_list)
        # ---Load the pre-generated conformal blocks for short multiplets---
        blocks_short = utils.generate_block_list_npy('6d/6d_blocks_short', [''], params.z_kill_list)
        # ---Load the pre-generated conformal blocks for long multiplets---
        blocks_long = utils.generate_block_list_npy('6d/6d_blocks_spin',
                                                    np.unique(self.spin_list_long).tolist(),
                                                    params.z_kill_list)
        block_array = np.array([blocks_chi[0], blocks_short[0], blocks_long], dtype=object)
        # need to specify dtype=object in block_array to avoid deprecation warning
        #
        self.block_chi = block_array[0]
        self.block_list_short = block_array[1]
        self.block_list_long = block_array[2]

        self.lb_deltas = params.lb_deltas
        self.ub_deltas = params.ub_deltas

        self.lb_opecoeffs = params.lb_opecoeffs
        self.ub_opecoeffs = params.ub_opecoeffs

        self.lb = np.concatenate((self.lb_deltas, self.lb_opecoeffs))
        self.ub = np.concatenate((self.ub_deltas, self.ub_opecoeffs))
        self.bounds = (self.lb, self.ub)

        self.include_tail = params.include_tail

        self.INHO_VALUE = self.inhomo_z_vector() - self.include_tail * self.tail_contribution()
        self.SHORT_D_HYPERS = self.short_coeffs_d_multiplet()
        self.SHORT_B_HYPERS = self.short_coeffs_b_multiplet_array()

        self.same_spin_hierarchy_deltas = params.same_spin_hierarchy  # impose weight separation flag
        self.partition = 1 + np.where(np.diff(self.spin_list_long) != 0)[0]  # this is where the spins increase
        self.dyn_shifts = np.split(params.dyn_shifts, self.partition)

    def h(self, z):
        """Computes h(z) defined in eqn (3.21) of 1507.05637."""
        res_a = - ((1 / 3) * (z ** 3) - (z - 1) ** (-1) - (z - 1) ** (-2) - (1 / 3) * (z - 1) ** (-3) - z ** (-1)) \
                - 8 * self.coupling * (z - (z - 1) ** (-1) + np.log(1 - z)) - (1 / 6) + 8 * self.coupling
        return res_a

    def b_l(self, ell):
        """Computes the b_l coefficients given in (4.9) of 1507.05637."""
        half_ell = ell // 2  # This is needed to avoid error in factorial2 as it doesn't like float variables

        numerator1 = (ell + 1) * (ell + 3) * (ell + 2) ** 2 * sc.factorial(half_ell) \
                     * sc.factorial2(half_ell + 2, exact=True) * sc.factorial2(half_ell + 3, exact=True) \
                     * sc.factorial2(ell + 5, exact=True)
        numerator2 = self.coupling * 8 * (2 ** (-(half_ell + 1))) * (ell * (ell + 7) + 11) \
                     * sc.factorial2(ell + 3, exact=True) * sc.gamma(half_ell + 2)
        denominator1 = 18 * sc.factorial2(ell + 2, exact=True) * sc.factorial2(2 * ell + 5, exact=True)
        denominator2 = sc.factorial2(2 * ell + 5, exact=True)
        res_a = numerator1 / denominator1 + numerator2 / denominator2
        return res_a

    def a_chi(self):
        """Computes a truncated version of the function given in (4.11) of 1507.05637."""
        res_a = np.sum((2 ** k) * self.b_l(k) * self.block_chi[k // 2] for k in range(0, self.ell_max + 2, 2))
        # sum runs over even spins to upto (and including) ell_max.
        return res_a

    def tail_contribution(self):
        """Not yet implemented. Returns zeros."""
        return np.zeros(self.env_dim)

    def inhomo_z_vector(self):
        """
        Computes (a truncated version of) the RHS of (4.13) in 1507.05637 for all points in the z-sample
        simultaneously.
        """
        res_inhomo_z_vector = - ((self.z - self.z_conj) ** (-3)) * (
                (self.h(1 - self.z_conj) - self.h(1 - self.z)) * (((self.z - 1) * (self.z_conj - 1)) ** (-1))
                + (self.h(self.z_conj) - self.h(self.z)) * ((self.z * self.z_conj) ** (-1))) + self.a_chi()
        return res_inhomo_z_vector.real

    def short_coeffs_b_multiplet(self, ell):
        """
        Computes the contribution to the crossing equation of a single spin ell B multiplet for all points in the
        z-sample simultaneously.

        Parameters
        __________
        ell : int
            The even spin of the B[0,2] multiplet.

        Returns
        -------
        ndarray(env_dim,)

        """
        res = self.block_list_short[ell // 2]
        return res

    def short_coeffs_b_multiplet_array(self):
        """
        Aggregates the B[0,2] multiplet contributions of differing even spins together into an array.

        Returns
        -------
        ndarray(spin_list_short_b.size, env_dim)

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
        Computes the contribution to the crossing equation of a
        single spin D[0,4] multiplet for all points in the z-sample
        simultaneously.

        Returns
        -------
        ndarray(env_dim,)

        """
        return self.block_list_short[0]

    def long_cons(self, delta, ell):
        """
        Computes the (pregenerated) contribution to the crossing equation of a single
        even spin long multiplet for all points in the z-sample simultaneously.

        Parameters
        ----------
        delta : float
            The conformal weight of the long multiplet.
        ell : int
            The spin of the long multiplet.
        Returns
        -------
        ndarray(env_dim,)

        """
        ell = ell // 2  # make sure the spin is even
        # we clip delta to ensure that it lies within the range accessible in the pregenerated blocks
        delta = np.clip(delta,
                        a_min=self.delta_start[ell],
                        a_max=self.delta_start[ell] + self.delta_end_increment - self.delta_sep)
        delta_lattice_point = (delta - self.delta_start[ell]) / self.delta_sep
        lattice_lower = np.floor(delta_lattice_point).astype(np.uint32)
        # get the appropriate contribution from block_list_long based on spin, lattice point and
        # using linear interpolation of the blocks
        long_c = self.block_list_long[ell][lattice_lower] \
                 + (delta_lattice_point - lattice_lower) \
                 * (self.block_list_long[ell][1 + lattice_lower] - self.block_list_long[ell][lattice_lower])
        # we need to transpose to return a shape compatible with the short multiplet contributions
        return np.transpose(long_c)

    def long_coeffs_array(self, deltas):
        """
        Aggregates all the long multiplet contributions together into a single array.

        Returns
        -------
        long_c : ndarray(num_of_long, env_dim)

        """
        long_c = self.long_cons(deltas[0], self.spin_list_long[0])
        for x in range(1, deltas.size):
            long_c = np.vstack((long_c, self.long_cons(deltas[x], self.spin_list_long[x])))
        return long_c

    def split_cft_data(self, cft_data):
        """
        Splits an array of cft data into conformal weights
        and OPE-squared coefficients.

        Parameters
        ----------
        cft_data : array_like

        Returns
        -------
        deltas : array_like
                 A 1-D array of conformal weights of the L[0,0] multiplets.
        ope_d : array_like
                A 1-D array of OPE coefficients of the D[0,4] multiplet.
        ope_b : array_like
                A 1-D array of OPE coefficients of the B[0,2] multiplets.
        ope_long : array_like
                A 1-D array of OPE coefficients of the L[0,0] multiplets.

        """
        deltas = cft_data[:self.num_of_long]  # these are the conformal weights of long multiplets only
        ope_coeffs = cft_data[self.num_of_long:]  # opes of all multiplets
        ope_d = ope_coeffs[0:1]  # assumes there's always a single D-multiplet
        ope_b = ope_coeffs[1:self.num_of_short]
        ope_long = ope_coeffs[self.num_of_short:]
        return deltas, ope_d, ope_b, ope_long

    def impose_weight_separation(self, deltas):
        """
        Enforces minimum conformal weight separations between degenerate spin operators.
        The separations are specified in the attribute `dyn_shifts`.

        Parameters
        ----------
        deltas : ndarray

        Returns
        -------
        deltas_clipped : ndarray

        """
        split_deltas = np.split(deltas, self.partition)  # split deltas into same spin degeneracies
        for (split_delta, dyn_shift) in zip(split_deltas, self.dyn_shifts):
            if len(split_delta) > 1:  # only impose if there's more than a single degenerate operator
                for i in range(1, len(split_delta)):
                    # minimum value is previous value plus current dyn_shift
                    split_delta[i] = np.clip(split_delta[i], a_min=(split_delta[i - 1] + dyn_shift[i]), a_max=None)

        # imposing the separations shouldn't take us outside the user specified bounds, so we clip here
        deltas_clipped = np.clip(np.concatenate(split_deltas), a_min=self.lb_deltas, a_max=self.ub_deltas)

        return deltas_clipped

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
        cft_data_modified : ndarray
            A list of possibly modified CFT data.

        """
        deltas, ope_coeff_d, ope_coeff_b, ope_coeff_long = self.split_cft_data(cft_data)

        if self.same_spin_hierarchy_deltas:
            deltas = self.impose_weight_separation(deltas)

        short_cons_d_multiplet = ope_coeff_d * self.SHORT_D_HYPERS
        # broadcast the reshaped B multiplet ope coefficients over their crossing contributions
        short_cons_b_multiplet = ope_coeff_b.reshape(-1, 1) * self.SHORT_B_HYPERS
        # broadcast the reshaped long multiplet ope coefficients over their crossing contributions
        long_cons = ope_coeff_long.reshape(-1, 1) * self.long_coeffs_array(deltas)
        # long_cons.shape = (num_of_long, env_dim)

        # add up all the components
        constraints = self.INHO_VALUE + short_cons_d_multiplet + short_cons_b_multiplet.sum(axis=0) \
                      + long_cons.sum(axis=0)  # the .sum implements summation over multiplet spins
        # reconstruct the possibly modified cft data into single array
        cft_data_modified = np.concatenate((deltas, ope_coeff_d, ope_coeff_b, ope_coeff_long))
        return constraints, cft_data_modified


class CrossingTwoD:
    """Computes truncated crossing equation for 2d c=1 compactified boson
    detailed in 2108.09330.
    """

    def __init__(self, params, z_data):
        """

        Parameters
        ----------
        params : ParametersTwoD_cft
            An instance of the `ParametersTwoD_cft` class.
        z_data : ZData
            An instance of the `ZData` class

        """

        suffixes = np.arange(params.delta_end + 1, dtype=int)
        self.ga_list = utils.generate_block_list_npy('2d/2d_blocks_ga_s', suffixes, params.z_kill_list)
        # ga are s-channel blocks
        self.gb_list = utils.generate_block_list_npy('2d/2d_blocks_gb_s', suffixes, params.z_kill_list)
        # gb are t-channel blocks
        self.block_list = np.array([self.ga_list, self.gb_list])

        self.env_dim = z_data.env_dim
        self.z = z_data.z
        self.z_conj = z_data.z_conj

        self.coupling = params.hh  # this is half the conformal weight of the external operator
        self.spins = params.spin_list
        self.action_space_N = 2 * self.spins.size

        self.delta_start = params.delta_start
        self.delta_sep = params.delta_sep
        self.delta_end = params.delta_end
        self.delta_minimums = self.get_delta_minimums()

        self.block_type = params.block_type

        self.lb_deltas = params.lb_deltas  # lower bound array for the conformal weights
        self.ub_deltas = params.ub_deltas  # upper bound array for the conformal weights

        self.lb_opecoeffs = params.lb_opecoeffs
        self.ub_opecoeffs = params.ub_opecoeffs

        self.lb = np.concatenate((self.lb_deltas, self.lb_opecoeffs))
        self.ub = np.concatenate((self.ub_deltas, self.ub_opecoeffs))
        self.bounds = (self.lb, self.ub)

        self.include_tail = params.include_tail

        self.T_CHANNEL_Z_PREFACTOR = self.norm_power(self.z, self.z_conj, 2 * self.coupling)
        self.S_CHANNEL_Z_PREFACTOR = self.norm_power(1 - self.z, 1 - self.z_conj, 2 * self.coupling)

        self.partition = 1 + np.where(np.diff(self.spins) != 0)[0]  # this is where the spins increase
        self.same_spin_hierarchy_deltas = params.same_spin_hierarchy
        self.dyn_shifts = np.split(params.dyn_shifts, self.partition)

    def get_delta_minimums(self):
        delta_minimums = np.zeros(self.spins.size)
        for position, spin in enumerate(self.spins):
            delta_minimums[position] = self.delta_start[spin]
        return delta_minimums

    def crossing(self, cft_data):
        block_array = np.zeros((self.spins.size, self.env_dim))

        deltas, ope_coeffs = self.split_cft_data(cft_data)
        if self.same_spin_hierarchy_deltas:
            deltas = self.impose_weight_separation(deltas)

        deltas_clipped = np.clip(deltas,
                                 a_min=self.delta_minimums,
                                 a_max=self.delta_minimums + self.delta_end - self.delta_sep)
        lattice_points = np.rint((deltas_clipped - self.delta_minimums) / self.delta_sep).astype(np.uint32)
        # uint32 range is 0 to 4294967295
        # TODO: implement linear interpolation here
        for index, channel in enumerate(self.block_type):
            if channel == 's1' or channel == 's2':
                block_array[index] = self.S_CHANNEL_Z_PREFACTOR \
                                     * self.block_list[0, self.spins[index], lattice_points[index], :].T
            else:
                block_array[index] = - self.T_CHANNEL_Z_PREFACTOR \
                                     * self.block_list[1, self.spins[index], lattice_points[index], :].T

        combined = ope_coeffs.reshape(-1, 1) * block_array
        cons = combined.sum(axis=0)
        return cons, np.concatenate((deltas, ope_coeffs))

    def norm_power(self, z, z_conj, exponent):
        return (z * z_conj).real ** exponent

    def split_cft_data(self, cft_data):
        deltas = cft_data[:int(self.action_space_N / 2)]
        ope_coeffs = cft_data[int(self.action_space_N / 2):]
        return deltas, ope_coeffs

    def impose_weight_separation(self, deltas):
        split_deltas = np.split(deltas, self.partition)  # split deltas by "spin"
        for (split_delta, dyn_shift) in zip(split_deltas, self.dyn_shifts):
            if len(split_delta) > 1:
                for i in range(1, len(split_delta)):
                    split_delta[i] = np.clip(split_delta[i], a_min=(split_delta[i - 1] + dyn_shift[i]), a_max=None)

        # imposing the separations shouldn't take us outside the user specified bounds, so we clip here
        deltas_clipped = np.clip(np.concatenate(split_deltas), a_min=self.lb_deltas, a_max=self.ub_deltas)

        return deltas_clipped

    # def cons_ising(self, z_i, deltas, ope_coeffs, hh, block_type, spins):
    #     z = self.z_points[z_i, 0] * np.exp(self.z_points[z_i, 1] * 1.0j)
    #     zb = self.z_points[z_i, 0] * np.exp(-self.z_points[z_i, 1] * 1.0j)
    #     c_tmp = 0.0
    #     c_tmp2 = 0.0
    #     c = - (z ** (2 * hh)) * (zb ** (2 * hh)) + ((z-1) ** (2 * hh)) * ((zb-1) ** (2 * hh))
    #     c_abs = abs((z ** (2 * hh)) * (zb ** (2 * hh)).real) + abs(((z-1) ** (2 * hh)) * ((zb-1) ** (2 * hh)).real)
    #     for i in range(len(deltas)):
    #         if block_type[i] == 's1':
    #             c_tmp = ope_coeffs[i] * self.rounder(deltas[i], spins[i], block_type[i], z_i, i)
    #             c_tmp2 = - ope_coeffs[i] * self.rounder(deltas[i], spins[i], 't1', z_i, i)
    #             c += c_tmp + c_tmp2
    #             c_abs += abs(c_tmp.real) + abs(c_tmp2.real)
    #         if block_type[i] == 's2':
    #             c_tmp = ope_coeffs[i] * self.rounder(deltas[i], spins[i], block_type[i], z_i, i)
    #             c_tmp2 = - ope_coeffs[i] * self.rounder(deltas[i], spins[i], 't2', z_i, i)
    #             c += c_tmp + c_tmp2
    #             c_abs += abs(c_tmp.real) + abs(c_tmp2.real)
    #
    #     return [c.real, c_abs]


class CrossingOneD:
    """Computes a truncated crossing equation for the
    1d Wilson loop defect SCFT as detailed in 2107.08510 and 2203.09556.

    """

    def __init__(self, params, z_data):
        """
        Parameters
        ----------
        params : ParametersOneD_cft
            An instance of the `ParametersOneD_cft` class.
        z_data : ZData
            An instance of the `ZData` class

        """

        suffixes = [str(i) + '_to_' + str(i + 1) for i in range(params.delta_end)]
        block_list = utils.generate_block_list_npy('1d/1D_blocks_delta_', suffixes, params.z_kill_list)
        self.blocks = np.vstack([block for block in block_list])  # combine all arrays in block_list into a single array
        # note that there is a singularity at Delta=1,
        # you should avoid that data point by setting apropriate lower bounds.
        del block_list  # release some memory

        self.env_dim = z_data.env_dim
        self.z = z_data.z

        self.coupling = params.g  # the value of the cft coupling constant
        self.spins = params.spin_list_long  # the weak coupling values of J for the cft operators
        self.action_space_N = 2 * self.spins.size  # the number of unknown cft data to be solved for

        self.delta_start = params.delta_start  # the value of the lowest conformal weight in `blocks`
        self.delta_end = params.delta_end  # the value of the highest conformal weight in `blocks`
        self.delta_sep = params.delta_sep  # the step size in `blocks`

        self.lb_deltas = params.lb_deltas  # lower bound array for the conformal weights
        self.ub_deltas = params.ub_deltas  # upper bound array for the conformal weights

        self.lb_opecoeffs = params.lb_opecoeffs
        self.ub_opecoeffs = params.ub_opecoeffs

        self.lb = np.concatenate((self.lb_deltas, self.lb_opecoeffs))
        self.ub = np.concatenate((self.ub_deltas, self.ub_opecoeffs))
        self.bounds = (self.lb, self.ub)

        # flag to impose minimum weight separation between degenerate spin operators
        self.same_spin_hierarchy_deltas = params.same_spin_hierarchy
        # array of positions where the spins increase
        self.partition = 1 + np.where(np.diff(self.spins) != 0)[0]
        # minimum weight separations split into degenerate spin families
        self.dyn_shifts = np.split(params.dyn_shifts, self.partition)

        self.include_tail = params.include_tail  # flag to include the coupling=0 analytic contribution to crossing eqn

        # compute the vector of fixed contributions to the crossing eqn
        self.G_SIMPLE = self.g_simple_contribution(self.coupling) - self.include_tail * self.tail_contribution()

    def F_I(self, x):
        """Compute F_I(x) defined in eqn (2.26) of 2203.09556."""
        res = x
        return res

    def F_B2(self, x):
        """Compute F_B2(x) defined in eqn (2.27) of 2203.09556."""
        res = x - x * sc.hyp2f1(1, 2, 4, x)
        return res

    def C2BPS(self, g):
        """Computes the analytical OPE-squared of B_2 multiplet
        given in eqn (2.29) of 2203.09556.

        Parameters
        ----------
        g : float
            The value of the coupling constant.

        Returns
        -------
        res : float
              The value of the OPE-squared coefficient evaluated at `coupling`.

        """
        if g == 0:
            return -1
        else:
            denominator = 2 * (g ** 2) * (np.pi ** 2) * (sc.iv(2, 4 * g * np.pi) ** 2)
            numerator = 3 * sc.iv(1, 4 * g * np.pi) * ((2 * (g ** 2) * (np.pi ** 2) + 1) * sc.iv(1, 4 * g * np.pi)
                                                       - 2 * g * np.pi * sc.iv(0, 4 * g * np.pi))
            res = - 1 + numerator * (denominator ** (-1))
            return res

    def crossed_F_I(self, x):
        """ Compute value of the crossed :math:`F_I`
        conformal block given in eqn (2.26) and eqn (2.32) of 2203.09556.

        Parameters
        ----------
        x : float
            The cross-ratio coordinate that the block is evaluated at.

        Returns
        -------
        res : ndarray
            The value of the crossed conformal block evaluated at `x`.

        """
        res = ((1 - x) ** 2) * self.F_I(x) + (x ** 2) * self.F_I(1 - x)
        return res

    def crossed_F_B2(self, x):
        """Compute crossed F_B2 block."""
        res = ((1 - x) ** 2) * self.F_B2(x) + (x ** 2) * self.F_B2(1 - x)
        return res

    def g_simple_contribution(self, g):
        """Computes the simple part of eqn (2.31) in 2203.09556
        for each point in the cross-ratio sampling.

        """
        res = self.crossed_F_I(self.z) + self.C2BPS(g) * self.crossed_F_B2(self.z)
        return res

    def split_cft_data(self, cft_data):
        """Splits an array of cft data into conformal weights
        and OPE-squared coefficients.

        Parameters
        ----------
        cft_data : ndarray

        Returns
        -------
        deltas : ndarray
                 A 1-D array of conformal weights.
        ope_coeffs : ndarray
                     A 1-D array of OPE coefficients.
        """
        deltas = cft_data[:self.spins.size]
        ope_coeffs = cft_data[self.spins.size:]
        return deltas, ope_coeffs

    def impose_weight_separation(self, deltas):
        """ Enforces minimum conformal weight separations between degenerate spin operators.
        The separations are specified in the attribute `dyn_shifts`.

        Parameters
        ----------
        deltas : ndarray

        Returns
        -------
        deltas_clipped : ndarray

        """
        split_deltas = np.split(deltas, self.partition)  # split deltas into same spin degeneracies
        for (split_delta, dyn_shift) in zip(split_deltas, self.dyn_shifts):
            if len(split_delta) > 1:  # only impose if there's more than a single degenerate operator
                for i in range(1, len(split_delta)):
                    # minimum value is previous value plus current dyn_shift
                    split_delta[i] = np.clip(split_delta[i], a_min=(split_delta[i - 1] + dyn_shift[i]), a_max=None)

        # imposing the separations shouldn't take us outside the user specified bounds, so we clip here
        deltas_clipped = np.clip(np.concatenate(split_deltas), a_min=self.lb_deltas, a_max=self.ub_deltas)

        return deltas_clipped

    def c(self, j):
        """
        Computes the coefficient in eqn (5.30) of 2203.09556.

        Parameters
        ----------
        j: float
            The weak coupling "spin".
        """

        return 0.25 ** (j + 1) * np.sqrt(np.pi) * (j - 1) * sc.gamma(j + 3) / sc.gamma(j + 1.5)

    def tail_contribution(self):
        """ Computes the E* contribution at the exact g=0 CFT data"""

        tail_pregen = self.g_simple_contribution(0)
        for spin in np.unique(self.spins):
            if spin == 1:
                tail_pregen += - (1 / 2) * (self.z ** 2) * ((1 - self.z) ** 2) \
                               * (sc.hyp2f1(2, 3, 6, self.z) + sc.hyp2f1(2, 3, 6, 1 - self.z))
            else:
                tail_pregen += self.c(int(spin)) * self.blocks[int(np.rint((spin - self.delta_start) / self.delta_sep))]
        return tail_pregen

    def crossing(self, cft_data):
        """Computes vector of crossing equations at different z-points
        corresponding to eqn (2.31) of 2203.09556.

        Parameters
        ----------
        cft_data : ndarray
                   An array of cft data at which the crossing equation is to be evaluated.

        Returns
        -------
        crossing_violations : ndarray
                              A 1-D array containing the evaluated values of the crossing equation.
                              The array has size `env_dim` and each index corresponds
                              to a single cross-ratio value.
        cft_data_modified : ndarray
                            A 1-D array of size `action_space_N`.

        """
        deltas, ope_coeffs = self.split_cft_data(cft_data)

        if self.same_spin_hierarchy_deltas:
            deltas = self.impose_weight_separation(deltas)

        # translate the delta array into
        delta_lattice_points, lattice_lower = self.weights_to_lattice(deltas)

        # perform linear interpolation of the pregenerated :math:`F_\Delta` crossed conformal blocks
        blocks_interpolated = self.blocks[lattice_lower] \
                              + (delta_lattice_points - lattice_lower).reshape(-1, 1) \
                              * (self.blocks[1 + lattice_lower] - self.blocks[lattice_lower])

        # broadcast the OPE coefficients over the interpolated blocks to get array of :math:`C_n^2 G_{\Delta_n}`
        g_delta_contributions = ope_coeffs.reshape(-1, 1) * blocks_interpolated
        # perform the sum over n in eqn (2.31) and add the :math:`G_{simple}` contribution
        crossing_violations = g_delta_contributions.sum(axis=0) + self.G_SIMPLE
        # reconstruct the possibly modified cft data into single array
        cft_data_modified = np.concatenate((deltas, ope_coeffs))
        return crossing_violations, cft_data_modified

    def weights_to_lattice(self, deltas):
        """Converts array of conformal weight values into
        lattice positions to allow easy indexing.

        The weights get clipped to lie within the range defined
        via `delta_start`, `delta_end` and `delta_sep`.

        Parameters
        ----------
        deltas : ndarray
                A 1-D array of floats representing values of conformal weights.

        Returns
        -------
        lattice_points : ndarray
                        A 1-D array of floats.
        lattice_lower : ndarray
                       `lattice_points` rounded down to integer values.

        """
        # we clip the deltas so that they are within the range that we have pre-generated
        deltas_clipped = np.clip(deltas,
                                 a_min=self.delta_start,
                                 a_max=self.delta_start + self.delta_end - self.delta_sep)
        lattice_points = (deltas_clipped - self.delta_start) / self.delta_sep  # these are floats
        lattice_lower = np.floor(lattice_points).astype(np.uint32)  # rounded down to integer
        # uint32 range is 0 to 4294967295 so more than sufficient to contain pre-generated lattice
        return lattice_points, lattice_lower


class CrossingOneD_deriv:
    """
    Computes a truncated, differentiated, crossing equation for the
    1d Wilson loop defect SCFT as detailed in 2107.08510 and 2203.09556.

    """

    def __init__(self, params):
        """
        Parameters
        ----------
        params : ParametersOneD_cft
                An instance of the `ParametersOneD_cft` class.

        """

        suffixes = [str(i) + '_to_' + str(i + 1) for i in range(params.delta_end)]
        block_list = utils.generate_block_list_npy('1d derivs/1d_deriv_normalised_delta_', suffixes, params.z_kill_list)
        self.blocks = np.vstack([block for block in block_list])  # combine all arrays in block_list into a single array
        del block_list
        self.env_dim = self.blocks.shape[1]  # this is 0.5 * N_der in 2203.09556

        self.coupling = params.g  # the value of the cft coupling constant
        self.spins = params.spin_list_long  # the weak coupling values of J for the cft operators
        self.action_space_N = 2 * self.spins.size  # the number of unknown cft data to be solved for

        self.delta_start = params.delta_start  # the value of the lowest conformal weight in `blocks`
        self.delta_end = params.delta_end  # the value of the highest conformal weight in `blocks`
        self.delta_sep = params.delta_sep  # the step size in `blocks`

        self.lb_deltas = params.lb_deltas  # lower bound array for the conformal weights
        self.ub_deltas = params.ub_deltas  # upper bound array for the conformal weights

        self.lb_opecoeffs = params.lb_opecoeffs
        self.ub_opecoeffs = params.ub_opecoeffs

        self.lb = np.concatenate((self.lb_deltas, self.lb_opecoeffs))
        self.ub = np.concatenate((self.ub_deltas, self.ub_opecoeffs))
        self.bounds = (self.lb, self.ub)

        # flag to impose minimum weight separation between degenerate spin operators
        self.same_spin_hierarchy_deltas = params.same_spin_hierarchy
        # array of positions where the spins increase
        self.partition = 1 + np.where(np.diff(self.spins) != 0)[0]
        # minimum weight separations split into degenerate spin families
        self.dyn_shifts = np.split(params.dyn_shifts, self.partition)

        self.include_tail = params.include_tail  # flag to include the coupling=0 analytic contribution to crossing eqn

        # compute the vector of fixed contributions to the crossing eqn
        self.G_SIMPLE = self.g_simple_contribution(self.coupling) - self.include_tail * self.tail_contribution()

    def C2BPS(self, g):
        """Computes the analytical OPE-squared of B_2 multiplet
        given in eqn (2.29) of 2203.09556.

        Parameters
        ----------
        g : float
            The value of the coupling constant.

        Returns
        -------
        res : float
              The value of the OPE-squared coefficient evaluated at `coupling`.

        """
        if g == 0:
            return 1
        else:
            denominator = 2 * (g ** 2) * (np.pi ** 2) * (sc.iv(2, 4 * g * np.pi) ** 2)
            numerator = 3 * sc.iv(1, 4 * g * np.pi) * ((2 * (g ** 2) * (np.pi ** 2) + 1) * sc.iv(1, 4 * g * np.pi)
                                                       - 2 * g * np.pi * sc.iv(0, 4 * g * np.pi))
            res = - 1 + numerator * (denominator ** (-1))
            return res

    def crossed_F_I_deriv(self):
        """Compute even order derivatives of the crossed :math:`F_I`
        conformal block given in eqn (2.26) and eqn (2.32) of 2203.09556.

        Returns
        -------
        F_I_derivs : ndarray
                     Returns a 1-D array of size `env_dim`

        """
        if self.env_dim == 1:
            result = [0.25]
        elif self.env_dim == 2:
            result = [0.25, -0.25]
        else:
            result = [0.25, -0.25] + [0.0] * (self.env_dim - 2)
        F_I_derivs = np.array(result)
        return F_I_derivs

    def crossed_F_B2_deriv(self):
        """Compute even order derivatives of the crossed :math:`F_{B_2}`
        conformal block given in eqn (2.27) and eqn (2.32) of 2203.09556.

        Returns
        -------
        F_B2_derivs : ndarray
                     Returns a 1-D array of size `env_dim`

        """

        def inter_2(n):
            # valid for n>4
            n = mp.mpf(n)
            numerator = 6 * (2 + 11 * n - 16 * n ** 2 + 8 * n ** 3)
            demoninator = n * (n - 1) * (n - 2)
            res_inter_2 = - 24 + numerator / demoninator \
                          + 12 * (2 * n - 1) * (mp.harmonic(0.5 * n - 2.0) - mp.harmonic(0.5 * n - 1.5))
            return float(res_inter_2)

        if self.env_dim == 1:
            result = [-4.25 + 6 * np.log(2)]
        elif self.env_dim == 2:
            result = [-4.25 + 6 * np.log(2), -49.75 + 72 * np.log(2)]
        elif self.env_dim == 3:
            result = [-4.25 + 6 * np.log(2), -49.75 + 72 * np.log(2), -116.5 + 168 * np.log(2)]
        else:
            result = [-4.25 + 6 * np.log(2), -49.75 + 72 * np.log(2), -116.5 + 168 * np.log(2)] \
                     + [inter_2(2 * n) for n in range(3, self.env_dim, 1)]
        F_B2_derivs = np.array(result)
        return F_B2_derivs

    def g_simple_contribution(self, g):
        """Computes even order derivatives of :math:`G_{simple}(g,x)` given
        in eqn (2.31) in 2203.09556.

        Parameters
        ----------
        g : float
            The value of the coupling constant.
        Returns
        -------
        res : ndarray
              Returns a 1-D array of size `env_dim`.

        """
        res = self.crossed_F_I_deriv() + self.C2BPS(g) * self.crossed_F_B2_deriv()
        return res

    def split_cft_data(self, cft_data):
        """Splits an array of cft data into conformal weights
        and OPE-squared coefficients.

        Parameters
        ----------
        cft_data : ndarray

        Returns
        -------
        deltas : ndarray
                 A 1-D array of conformal weights.
        ope_coeffs : ndarray
                     A 1-D array of OPE coefficients.

        """
        deltas = cft_data[:self.spins.size]
        ope_coeffs = cft_data[self.spins.size:]
        return deltas, ope_coeffs

    def impose_weight_separation(self, deltas):
        """Enforces minimum conformal weight separations between degenerate spin operators.
        The separations are specified in the attribute `dyn_shifts`.

        Parameters
        ----------
        deltas : ndarray
                A 1-D array of conformal weights
        Returns
        -------
        deltas_clipped : ndarray
                A 1-d array of conformal weights with minimum separations imposed.

        Notes
        _____
        The user specified bounds `lb_deltas` and `ub_deltas` can override the separations.

        """
        split_deltas = np.split(deltas, self.partition)  # split deltas into same spin degeneracies
        for (split_delta, dyn_shift) in zip(split_deltas, self.dyn_shifts):
            if len(split_delta) > 1:  # only impose if there's more than a single degenerate operator
                for i in range(1, len(split_delta)):
                    # minimum value is previous value plus current dyn_shift
                    split_delta[i] = np.clip(split_delta[i], a_min=(split_delta[i - 1] + dyn_shift[i]), a_max=None)

        # imposing the separations shouldn't take us outside the user specified bounds, so we clip here
        deltas_clipped = np.clip(np.concatenate(split_deltas), a_min=self.lb_deltas, a_max=self.ub_deltas)

        return deltas_clipped

    def c(self, j):
        """Computes the coefficient in eqn (5.30) of 2203.09556.

        Parameters
        ----------
        j: float
            The weak coupling "spin".

        """
        return 0.25 ** (j + 1) * np.sqrt(np.pi) * (j - 1) * sc.gamma(j + 3) / sc.gamma(j + 1.5)

    def tail_contribution(self):
        """Computes the E* contribution at the exact g=0 CFT data."""
        tail_pregen = self.g_simple_contribution(0)
        for spin in np.unique(self.spins):
            if spin == 1:
                # see discussion at bottom of page 36 of 2203.09556
                tail_pregen += - 0.5 * self.blocks[int(np.rint((spin - self.delta_start) / self.delta_sep))]
            else:
                tail_pregen += self.c(int(spin)) * self.blocks[int(np.rint((spin - self.delta_start) / self.delta_sep))]
        return tail_pregen

    def crossing(self, cft_data):
        """Computes vector of differentiated crossing equations
        corresponding to eqn (2.31) of 2203.09556.

        Parameters
        ----------
        cft_data : ndarray
                   An array of cft data at which the crossing equation is to be evaluated.

        Returns
        -------
        crossing_violations : ndarray
                              A 1-D array containing the evaluated values of the crossing equation.
                              The array has size `env_dim` and each index corresponds
                              to a single even order of derivatives
                              (see footnote 7 on page 16 of 2203.09556).
        cft_data_modified : ndarray
                            A 1-D array of size `action_space_N`.

        """
        deltas, ope_coeffs = self.split_cft_data(cft_data)

        if self.same_spin_hierarchy_deltas:
            deltas = self.impose_weight_separation(deltas)

        # translate the delta array into
        delta_lattice_points, lattice_lower = self.weights_to_lattice(deltas)

        # perform linear interpolation of the pregenerated :math:`F_\Delta` crossed conformal blocks
        blocks_interpolated = self.blocks[lattice_lower] \
                              + (delta_lattice_points - lattice_lower).reshape(-1, 1) \
                              * (self.blocks[1 + lattice_lower] - self.blocks[lattice_lower])

        # broadcast the OPE coefficients over the interpolated blocks to get array of :math:`C_n^2 G_{\Delta_n}`
        g_delta_contributions = ope_coeffs.reshape(-1, 1) * blocks_interpolated
        # perform the sum over n in eqn (2.31) and add the :math:`G_{simple}` contribution
        crossing_violations = g_delta_contributions.sum(axis=0) + self.G_SIMPLE
        # reconstruct the possibly modified cft data into single array
        cft_data_modified = np.concatenate((deltas, ope_coeffs))
        return crossing_violations, cft_data_modified

    def weights_to_lattice(self, deltas):
        """Converts array of conformal weight values into
        lattice positions to allow easy indexing.

        The weights get clipped to lie within the range defined
        via `delta_start`, `delta_end` and `delta_sep`.

        Parameters
        ----------
        deltas : ndarray
                A 1-D array of floats representing values of conformal weights.

        Returns
        -------
        lattice_points : ndarray
                        A 1-D array of floats.
        lattice_lower : ndarray
                       `lattice_points` rounded down to integer values.

        """
        # we clip the deltas so that they are within the range that we have pre-generated
        deltas_clipped = np.clip(deltas,
                                 a_min=self.delta_start,
                                 a_max=self.delta_start + self.delta_end - self.delta_sep)
        lattice_points = (deltas_clipped - self.delta_start) / self.delta_sep  # these are floats
        lattice_lower = np.floor(lattice_points).astype(np.uint32)  # rounded down to integer
        # uint32 range is 0 to 4294967295 so more than sufficient to contain pre-generated lattice
        return lattice_points, lattice_lower
