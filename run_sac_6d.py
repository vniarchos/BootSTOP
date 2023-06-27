import sys
import numpy as np
from configs.parametersSixD import ParametersSixD_cft, ParametersSixD_sac
from environment.cfts import CrossingSixD
from environment.data_z_sample import ZData_6d_250
from neural_net.sac import soft_actor_critic

if __name__ == '__main__':
    # ---Instantiating some relevant classes---
    # params_cft = ParametersSixD_cft()
    # params_sac = ParametersSixD_sac()
    params_cft = ParametersSixD_cft()
    params_sac = ParametersSixD_sac()
    zd = ZData_6d_250()
    zd.kill_data(params_cft.z_kill_list)
    cft = CrossingSixD(params_cft, zd)

    # array_index is the cluster array number passed to the console. Set it to zero if it doesn't exist.
    try:
        array_index = int(sys.argv[1])
    except IndexError:
        array_index = 0

    # form the file_name where the code output is saved to
    file_name = params_sac.filename_stem + str(array_index) + '.csv'

    # put together stuff
    guessing_run_list = np.concatenate((params_sac.guessing_run_list_deltas,
                                        params_sac.guessing_run_list_opes))
    guess_sizes = np.concatenate((params_sac.guess_sizes_deltas, params_sac.guess_sizes_opes))

    # determine initial starting point in the form needed for the soft_actor_critic function
    x0 = params_sac.global_best - cft.lb

    # ---Run the soft actor critic algorithm---
    soft_actor_critic(func=cft.crossing,
                      max_window_changes=params_sac.max_window_exp,
                      window_decrease_rate=params_sac.window_rate,
                      pc_max=params_sac.pc_max,
                      file_name=file_name,
                      array_index=array_index,
                      bounds=cft.bounds,
                      search_window_sizes=guess_sizes,
                      guessing_run_list=guessing_run_list,
                      environment_dim=cft.env_dim,
                      search_space_dim=cft.action_space_N,
                      faff_max=params_sac.faff_max,
                      starting_reward=params_sac.global_reward_start,
                      x0=x0,
                      verbose=params_sac.verbose)
