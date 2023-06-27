import pygmo as pg
import numpy as np


class udp_basic:
    """
    A basic unconstrained user defined problem which can handle any cft.
    The objective to be minimised is the norm of the vector of crossing equation
    values.

    Notes
    _____
    See https://esa.github.io/pygmo2/tutorials/coding_udp_simple.html

    """

    def __init__(self, cft):
        """
        Parameters
        ----------
        cft : callable
            An instance of one of the classes contained in environment.cfts.py
        """
        self.cft = cft

    def fitness(self, x):
        cons, _ = self.cft.crossing(x)
        obj = np.linalg.norm(cons)  # the objective to minimise

        return (obj,)

    def get_bounds(self):
        return self.cft.bounds

    def gradient(self, x):
        return pg.estimate_gradient(lambda y: self.fitness(y), x)


class udp_1d:
    """
    A pygmo user defined problem specific to the 1d defect cft with three inequality constraints.

    Notes
    _____
    See https://esa.github.io/pygmo2/tutorials/coding_udp_constrained.html

    The inequality constraints imposed on the solution are

    1) :math:`\Delta_2 - \Delta_3 \leq 0`.

    2) :math:`C_2^2 - 0.1 \leq 0`.

    3) :math:`0.1 - C_3^2 \leq 0`.

    """
    def __init__(self, cft):
        """
        Parameters
        ----------
        cft : callable
            An instance of either the class CrossingOneD
            or CrossingOneD_deriv contained in environment.cfts.py
        """
        self.cft = cft

    def fitness(self, x):
        cons, _ = self.cft.crossing(x)
        obj = np.linalg.norm(cons)  # the objective to minimise

        x_deltas, x_ope_coeffs = self.cft.split_cft_data(x)

        ineq1 = x_deltas[1] - x_deltas[2]
        ineq2 = x_ope_coeffs[1] - 0.1
        ineq3 = 0.1 - x_ope_coeffs[2]

        return (obj, ineq1, ineq2, ineq3)

    def get_bounds(self):
        return self.cft.bounds

    def gradient(self, x):
        return pg.estimate_gradient(lambda y: self.fitness(y), x)

    def get_nic(self):
        return 3


class udp_1d_integral_constraints:
    """
    A pygmo user defined problem specific to the 1d defect cft which
    imposes the two integral constraints of 2203.09556.

    Notes
    _____

    """
    def __init__(self, cft, int1, rhs1, int2, rhs2):
        """

        Parameters
        ----------
        cft : callable
            An instance of either the class CrossingOneD
            or CrossingOneD_deriv contained in environment.cfts.py
        int1 : array_like
            An array containing a range of conformal weights and
            the results of eqn (C.7) of 2203.09556 evaluated using those weights.
        rhs1 : array_like
            An array containing sample values of the cft coupling constant and
            the results of eqn (C.9) of 2203.09556 evaluated on those values.
        int2 : array_like
            An array containing a range of conformal weights and
            the results of eqn (C.3) of 2203.09556 evaluated using those weights.
        rhs2 : array_like
            An array containing sample values of the cft coupling constant and
            the results of eqn (C.4) of 2203.09556 evaluated on those values.
        """
        self.cft = cft
        self.int1_array = int1
        self.rhs1_array = rhs1
        self.int2_array = int2
        self.rhs2_array = rhs2

        # perform some simple consistency checks on the pre-generated data
        if not np.array_equal(self.int1_array[0], self.int2_array[0]):
            # int1_array[0], int2_array[0] are the conformal weights that
            # int1_array[1], int2_array[1] are computed at
            raise SystemExit('IntDataMismatch')

        if not np.array_equal(self.rhs1_array[0], self.rhs2_array[0]):
            # rhs1_array[0], rhs2_array[0] are the coupling values that
            # rhs1_array[1], rhs2_array[1] are computed at
            raise SystemExit('RhsDataMismatch')

        if self.cft.coupling not in self.rhs1_array[0]:
            # check if the cft coupling value is in rhs1_array[0]
            raise SystemExit('CouplingRhsDataMismatch')

        # Select the values of the integral constraint RHS which correspond to the cft coupling
        # The tail contribution to the first integral constraint is zero.
        self.CONSTRAINT_1_RHS = self.rhs1_array[1][np.where(self.rhs1_array[0] == self.cft.coupling)[0][0]]
        self.CONSTRAINT_2_RHS = self.rhs2_array[1][np.where(self.rhs2_array[0] == self.cft.coupling)[0][0]] \
                                - self.cft.include_tail * self.tail_contribution_to_constraint2()

    def fitness(self, x):
        cons, _ = self.cft.crossing(x)
        obj = np.linalg.norm(cons)

        # calculate the values of the integral constraints
        x_deltas, x_ope_coeffs = self.cft.split_cft_data(x)
        delta_lattice_points, lattice_lower = self.cft.weights_to_lattice(x_deltas)

        constraints = [self.CONSTRAINT_1_RHS, self.CONSTRAINT_2_RHS]

        working_block = [self.int1_array[1], self.int2_array[1]]

        for ind in [0, 1]:
            interpolated = working_block[ind][lattice_lower] \
                           + (delta_lattice_points - lattice_lower) \
                           * (working_block[ind][1 + lattice_lower] - working_block[ind][lattice_lower])

            constraints[ind] += np.sum(x_ope_coeffs * interpolated, axis=0)  # computes (C.2) and (C.6) of 2203.09556

        return [obj, constraints[0], constraints[1]]

    def tail_contribution_to_constraint2(self):
        """ Computes the E* contribution to the 2nd constraints at the exact g=0 CFT data"""
        tail_pregen = - 1 / 6  # this is (minus) the g=0 limit of rhs_2
        for spin in np.unique(self.cft.spins):
            if spin == 1:
                tail_pregen += 0.1518615277338802  # this is 85/12 - 10 log(2) to 16 d.p.

            else:
                tail_pregen += self.cft.c(int(spin)) \
                               * self.int2_array[1][int(np.rint((spin - self.cft.delta_start) / self.cft.delta_sep))]
        return tail_pregen

    def get_bounds(self):
        return self.cft.bounds

    def gradient(self, x):
        return pg.estimate_gradient(lambda y: self.fitness(y), x)

    def get_nec(self):
        return 2
