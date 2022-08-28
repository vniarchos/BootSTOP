import sys
from parameters import ParametersSixD_SAC
from environment.blocks import CrossingSixD_SAC
from environment.data_z_sample import ZData
import environment.utils as utils
from neural_net.sac import soft_actor_critic

if __name__ == '__main__':
    # ---Instantiating some relevant classes---
    params = ParametersSixD_SAC()
    zd = ZData()

    # ---Kill portion of the z-sample data if required---
    zd.kill_data(params.z_kill_list)

    # ---Load the pre-generated conformal blocks for long multiplets---
    blocks = utils.generate_block_list(max(params.spin_list_long), params.z_kill_list)

    # ---Instantiate the crossing_eqn class---
    cft = CrossingSixD_SAC(blocks, params, zd)

    # array_index is the cluster array number passed to the console. Set it to zero if it doesn't exist.
    try:
        array_index = int(sys.argv[1])
    except IndexError:
        array_index = 0

    # form the file_name where the code output is saved to
    file_name = params.filename_stem + str(array_index) + '.csv'

    # determine initial starting point in the form needed for the soft_actor_critic function
    x0 = params.global_best - params.shifts

    # ---Run the soft actor critic algorithm---
    soft_actor_critic(func=cft.crossing,
                      max_window_changes=params.max_window_exp,
                      window_decrease_rate=params.window_rate,
                      pc_max=params.pc_max,
                      file_name=file_name,
                      array_index=array_index,
                      lower_bounds=params.shifts,
                      search_window_sizes=params.guess_sizes,
                      guessing_run_list=params.guessing_run_list,
                      environment_dim=zd.env_shape,
                      search_space_dim=params.action_space_N,
                      faff_max=params.faff_max,
                      starting_reward=params.global_reward_start,
                      x0=x0,
                      verbose=params.verbose)
