# BootSTOP (Bootstrap STochastic OPtimizer)

---
<!-- TOC -->
* [BootSTOP (Bootstrap STochastic OPtimizer)](#bootstop--bootstrap-stochastic-optimizer-)
  * [Overview](#overview)
  * [Getting started](#getting-started)
  * [Running the Code](#running-the-code)
  * [Changing the setup](#changing-the-setup)
  * [Changing the pregenerated conformal blocks](#changing-the-pregenerated-conformal-blocks)
  * [Requirements](#requirements)
  * [References](#references)
<!-- TOC -->

## Overview
This code applies the soft-Actor-Critic algorithm to solve a truncated crossing equation 
for a conformal field theory (currently only the 6D (2,0) CFT). See references [1], [2] and [3] for comprehensive background.

## Getting started

In this readme file we will describe how to quickly get started with using the
code. First things first, you will need to follow this [link](https://drive.google.com/drive/folders/1myAXJzBhFl5eYI1iFimA4i5ukeinxzw5?usp=sharing)
to download the pregenerated conformal block csv files (each file is around 0.2Gb) and place them within the *block_lattices* 
folder.

## Running the Code

The main file for running the code is *run_sac.py*. Running it is simple, all you have to do is execute the following in the terminal:

`python run_sac.py array`

where `array` is an _optional_ integer which is incorporated in the csv filename 
which the code creates and outputs to. This flexibliity allows for the use of a cluster to 
run many instances of BootSTOP in parallel.
You might need to change "python" to "python3" depending on your installation of
Python.

## Changing the setup

Here is a list of parameters that must be specified by the user in order for BootSTOP to operate, they are all stored 
within classes in the *parameters.py* file.

Within the class *ParametersSixD*. These alter the conformal blocks.
- **inv_c_charge:** This is the inverse of the central charge of the CFT.
- **spin_list_short_d:** This is the spin of the D multiplet.
- **spin_list_short_b:** This is a list of spins for the B multiplets.
- **spin_list_long:** This is the list of spins for the unprotected long multiplets. 
- **ell_max:** The maximum spin cutoff for multiplets completely fixed by chiral algebra.
- **delta_start:** The lowest conformal weights used in the pregenerated conformal blocks. 
- **delta_end_increment:** Increment added to delta_start to identify the highest conformal weights used 
in the pregenerated conformal blocks.
- **delta_sep:** The regular spacing between conformal weights in the pregenerated conformal blocks.
- **z_kill_list:** Allows for a smaller z-point sample by specifying which points to remove.


Within the subclass *ParametersSixD_SAC*. These alter the soft-Actor-Critic behaviour.
- **filename_stem:** The name of the output file (note the filetype *.csv* is automatically appended along with an 
optional integer - see Running the Code).
- **verbose:** Controls how much output is printed to the python console.
- **faff_max:** The maximum number of iterations spent not improving the reward.
- **pc_max:** Maximum number of neural net reinitialisations before search window size decrease.
- **window_rate:** Multiplier between 0 and 1 applied to decrease the search window sizes.
- **max_window_exp:** The maximum number of search window size decreases.
- **same_spin_hierarchy:** Boolean flag to impose a separation in conformal weights of degenerate long multiplets.
- **dyn_shift:** Value for the separation of conformal weights of degenerate long multiplets.
- **guessing_run_list_deltas:** Set guessing status for individual unknown conformal weights.
- **guessing_run_list_opes:** Set guessing status for individual unknown OPE squared coefficients.
- **guess_sizes_deltas:** The initial sizes of the search windows for the conformal weights.
- **guess_sizes_opes:** The initial sizes of the search windows for the OPE squared coefficients.
- **shifts_deltas:** The lower bounds for the conformal weights.
- **shifts_opecoeffs:** The lower bounds for the OPE squared coefficients.
- **global_best:** Initial set of CFT data for algorithm to explore around.
- **global_reward_start:** Initial reward for algorithm to improve upon.

The sample of points in the complex plane is stored within *environment/data_z_sample.py*. Altering the entries 
contained within `zre` or `zim` would mean having to re-compute all the *block_lattices/6d_blocks_spin#.csv* files in 
order to avoid inconsistencies.

The hyperparameters which control the behaviour of the neural networks can be altered by specifying non-default 
parameter values when the Agent class in instantiated within the Learn class of *sac.py*. 

## Changing the pregenerated conformal blocks 

The pregenerated conformal blocks are computed using Mathematica in tandem with QMUL's Apocrita high performance
compute cluster. Here we describe how the spin 0 blocks were generated using the file 
*pregenerate_blocks/genblocks_6d_spin0.m*. Higher spins follow the same procedure 
with a suitable change of `spin` variable. 

First the choice of the sampling of the z-plane was made with the real and imaginary parts stored in the variables
`zre` and `zim`. Next a lower bound for the conformal weight is set within `floor`, we have consistently used 0.2 below 
the long multiplet unitarity bound so that the BootSTOP code can explore around that bound. Finally, the discretisation
of the conformal weights is set with `step` (we used 0.0005). The Mathematica code then loops evaluating the expression $$z \bar{z} a_{\Delta, \ell=0}^{\rm{at}} (z, \bar{z}) - (1-z)(1-\bar{z}) a_{\Delta, \ell=0}^{\rm{at}} (1-z, 1-\bar{z}),$$ where $a_{\Delta, \ell}^{\rm{at}}$ is defined in equation (4.5) of [arXiv:1507.05637](https://arxiv.org/pdf/1507.05637.pdf),
for all values of $z, \bar{z}$ in the z-sample with $\Delta$ starting at `floor` and increasing by `step` with 
each pass of the loop until it reaches `ceiling`. Once the loop is completed the output is exported to a csv file.
In practice setting `ceiling = floor - step + 1` with `step = 0.0005` allows the loop to complete in a reason amount
of time (~75mins) and a wide range for $\Delta$ (up to `floor + 30 - step`) can be built up by running in parallel on a 
cluster. The final, large, csv file *6d_blocks_spin0.csv* is formed by running the script 
*pregenerate_blocks/aggregate_block_data.py* with approriate values set for the variables.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- SciPy 1.8.0
- mpmath

## References

- [1] G. Kantor, V. Niarchos, C. Papageorgakis and P. Richmond,
*6D (2,0) Bootstrap with soft-Actor-Critic*, [arXiv:2209.xxxxx]
- [2] G. Kantor, V. Niarchos and C. Papageorgakis,
*Conformal bootstrap with reinforcement learning*,
Phys. Rev. D 105 (2022) [arXiv:2108.09330](https://arxiv.org/abs/2108.09330)
- [3] G. Kantor, V. Niarchos and C. Papageorgakis,
*Solving Conformal Field Theories with Artificial Intelligence*,
Phys. Rev. Lett. 128 (2022) [arXiv:2108.08859](https://arxiv.org/abs/2108.08859)
