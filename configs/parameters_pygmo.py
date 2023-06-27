class Parameters_pygmo:
    """Class used to hold parameters needed to initialise and customise a pygmo run."""

    def __init__(self):
        """
        udp : str
            String holding the name of the user defined problem.
            Must match the name of a class in environment.pygmo_udps.py.
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
        self.verbosity = 100
        self.output_file = '1d_pygmo_output.csv'
