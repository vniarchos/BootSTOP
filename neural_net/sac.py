import numpy as np
from neural_net.sac_agent import Agent
import environment.utils as utils


class ObjectiveFunWrapper:
    """
    Class used to wrap around the reward function.
    """
    def __init__(self, func, *args):
        self.func = func
        self.args = args

    def fun(self, x):
        return self.func(x, *self.args)


class Environment:
    """
    Class used to record the current location in parameter space as well as the move around the environment
    """

    def __init__(self, func_wrapper, environment_dim, search_space_dim,
                 lower_bounds, search_window_sizes, guessing_run_list):
        self.func_wrapper = func_wrapper
        self.environment_dim = environment_dim
        self.search_space_dim = search_space_dim
        self.lower_bounds = lower_bounds
        self.search_window_sizes = search_window_sizes
        self.guessing_run_list = guessing_run_list
        self.current_location = lower_bounds
        self.current_constraints = np.zeros(self.environment_dim, dtype=np.float32)
        self.current_reward = None
        self.reward_improved = False

    def move(self, action, largest, solution):
        """
        Function to implement moving in search space and evaluation of func_wrapper.fun at new location.

        Parameters
        ----------
        action : ndarray
            The neural net's current prediction. Values in range [-1,1].
        largest : float
            The largest reward value to compare against.
        solution : ndarray
            The solution to be used if in a non-guessing mode.
        """
        # updating location might be a theory dependent thing i.e. allowing some to be negative
        self.current_location = self.lower_bounds + abs(self.search_window_sizes * action
                                                        + (1 - self.guessing_run_list) * solution)

        # evaluate the function at the updated location and update attributes
        self.current_constraints, self.current_reward, self.current_location \
            = self.func_wrapper.fun(self.current_location)

        # check if we've found a new maximum
        if self.current_reward > largest:
            self.reward_improved = True

    def reset_env(self):
        """
        Resets the current_constraints, current_location and reward_improved attributes.
        """
        self.current_constraints = np.zeros(self.environment_dim, dtype=np.float32)
        self.current_location = self.lower_bounds
        self.reward_improved = False


class Learn:
    """
    Class that implements neural net learning.

    Parameters
    ----------
    env: Environment
        Instance of `Environment` class.
    """
    def __init__(self, env):
        self.env = env
        self.agent = Agent(input_dims=[env.environment_dim], n_actions=env.search_space_dim)
        self.done = False
        self.fdone = False
        self.observation = self.env.current_constraints
        self.observation_ = None
        self.step = 0
        self.faff = 0
        self.solution = None
        self.productivity_counter = None
        self.rewards = None

    def loop(self, window_scale_exponent, file_name, array_index, current_best, faff_max, verbose, output_order=''):
        """
        Loop which implements neural net learning and exploration of parameter space landscape.
        """
        self.solution = current_best
        self.productivity_counter = False
        while not self.fdone:
            # ---Running the learning loop---
            self.step += 1
            # get a prediction from the neural net based on observation
            action = self.agent.choose_action(self.observation)  # returns values in range [-1,1]
            # move to a new location based on neural net prediction and evaluate function there
            self.env.move(action, max(self.rewards), self.solution)
            self.observation_ = self.env.current_constraints  # the new obervation
            current_reward = self.env.current_reward
            current_location = self.env.current_location
            # update the neural net buffer with the latest data
            self.agent.remember(self.observation, action, current_reward, self.observation_, self.done)
            # learn from updated buffer
            self.agent.learn()
            self.observation = self.observation_  # update observation to be the current constraints
            # append the current reward to the list of previous rewards generated within the loop
            self.rewards.append(current_reward)

            if self.env.reward_improved:
                # solution is value of parameters above the lower bounds
                self.solution = np.copy(current_location - self.env.lower_bounds)
                # prepare some output to be saved
                if output_order == 'Reversed':
                    file_output = np.concatenate((current_location, current_reward, array_index), axis=None)
                else:
                    file_output = np.concatenate((array_index, current_reward, current_location), axis=None)
                # append the improved reward and current search space location to the csv file 'file_name'
                utils.output_to_file(file_name, file_output)
                # reset various quantities so that we're ready to go through the while loop again
                self.env.reset_env()
                self.faff = 0
                self.productivity_counter = True  # we have been productive in finding a better solution
            else:
                self.faff += 1  # we have not been productive so increase the faff counter

            # if we hit faff_max then flip the fdone flag to exit the while loop
            if self.faff == faff_max:
                self.fdone = True

            # form the output for the console
            console_output = 'step %.1f' % self.step, 'avg reward %.10f' % np.mean(self.rewards[-25:]), \
                             'current reward %.10f' % current_reward, 'max reward %.10f' % max(self.rewards), \
                             'zoom %.d' % window_scale_exponent, 'faff %.1f' % self.faff
            if verbose == 'e':  # print everytime we come through the loop
                utils.output_to_console(console_output)
            elif verbose == 'o' and self.fdone:  # print only if we are going to exit loop
                utils.output_to_console(console_output)
            else:
                pass  # don't print

        return self.solution


def soft_actor_critic(func,
                      environment_dim,
                      search_space_dim,
                      lower_bounds,
                      search_window_sizes,
                      pc_max,
                      window_decrease_rate,
                      max_window_changes,
                      faff_max,
                      file_name,
                      array_index,
                      guessing_run_list,
                      starting_reward,
                      x0,
                      verbose='',
                      args=()):
    """
    Apply the soft-Actor-Critic algorithm to a function.

    Parameters
    ----------
    func : callable
        The function to be maximised.
    environment_dim : int
        The number of points in the complex plane the function is evaluated at.
    search_space_dim : int
        The total number of parameters to be solved for.
    max_window_changes : int
        Maximum number of search window decreases.
    window_decrease_rate : float
        Search window size decrease rate. Value should be between 0 and 1.
    pc_max : int
        Maximum number of reinitialisations before search window decrease triggered.
    file_name : str
        The filename where output is to be saved.
    array_index : int
        An index which can be used to identify output data.
    starting_reward : float
        Initial reward value for SAC algorithm to try and beat.
    x0 : ndarray
        An array containing the initial solution to search around in parameter space.
    lower_bounds : ndarray
        An array containing absolute lower bounds on parameter search domain.
    search_window_sizes : ndarray
        An array containing initial search window sizes.
    guessing_run_list : ndarray
        An array containing Boolean values for guessing run or not.
    faff_max : int
        Maximum number of steps spent searching for an improved reward without success.
        When faff_max is reached a reinitialisation occurs.
    verbose : {'', 'e', 'o'}
        When the soft-Actor-Critic algorithm should print to the console:
        - ``: default value which produces no output.
        - `e`: print everytime reward is recalculated.
        - `o`: only after a reinitialisation.
    args : tuple, optional
        Any additional fixed parameters needed to completely specify the function.

    Notes
    -----
    Everytime a new highest reward is found it, and the location in parameter search space, are saved to a csv file.
    However, if a non-zero starting_reward is specified and no improvement is found then no output is saved.
    """

    # Wrapper for the reward function
    # the function needs to return an array of constraints, the reward and a (possibly) modified current_location
    func_wrapper = ObjectiveFunWrapper(func, *args)
    #
    environment = Environment(func_wrapper, environment_dim, search_space_dim, lower_bounds, search_window_sizes,
                              guessing_run_list)
    lrn = Learn(environment)
    # set the initial window size reduction exponent, pc counter and best_reward
    window_scale_exponent = 0
    pc = 0  # records number of neural net reinitialisations
    best_reward = starting_reward  # at the start the best reward is the starting reward

    # ---Looping until a certain window size is reached---
    while window_scale_exponent < max_window_changes:
        lrn.rewards = [best_reward]
        x0 = lrn.loop(window_scale_exponent, file_name, array_index, x0, faff_max, verbose=verbose)
        # update best_rewards as the maximum reward found after going through lrn.loop
        best_reward = max(lrn.rewards)
        if not lrn.productivity_counter:  # a better reward was not found by lrn.loop
            # increase the pc counter
            pc += 1
        if pc == pc_max:
            # put into a non-guessing run mode
            environment.guessing_run_list = np.zeros(search_space_dim, dtype=bool)
            # increase the window scale exponent and decrease the search window sizes
            window_scale_exponent += 1
            environment.search_window_sizes = window_decrease_rate * environment.search_window_sizes
            # reset the re-initialisation counter to zero
            pc = 0

        # delete and re-instantiate the Learn class, this re-initialises the Agent class
        del lrn
        environment.reset_env()
        lrn = Learn(environment)

    # when finished looping print the final reward and corresponding CFT data
    best_reward_location = x0 + lower_bounds
    utils.output_to_console('Maximum reward: %.16f' % best_reward)
    utils.output_to_console('Location of maximum reward: ')
    utils.output_to_console(best_reward_location.tolist())
