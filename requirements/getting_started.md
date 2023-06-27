# Getting Started

There are several ways to get started with the BootSTOP implementation.

We provide two recommended options: using Mamba or Anaconda. These options are preferable as they can handle dependencies for optimizers like IPOPT, which are not written in Python and cannot be installed with pip (see the [PyGMO docs](https://esa.github.io/pygmo2/install.html#pip) for more information).

## Option 1 - (Ana)conda or (Micro)mamba

Conda and Mamba offer a convenient way to create Python environments for different projects that may have different dependencies. We have created an environment called 'bootplug' with all the necessary dependencies pre-installed.

To get started, follow these installation instructions:

1. Make sure you have Mamba (recommended) or Conda installed. We recommend using Mamba as it is a faster reimplementation of Anaconda. You can find Mamba installation instructions on [Mamba's website](https://mamba.readthedocs.io/en/latest/installation.html). Alternatively, you can refer to the [Anaconda documentation](https://docs.anaconda.com/free/anaconda/install/index.html) for Anaconda installation instructions.

2. Install the environment from the provided `conda_env.yml` file:
   - If you're using Mamba (assuming micromamba), run the following command in your terminal, after navigating to the directory where the file is located:
     ```
     micromamba env create -f conda_env.yml
     ```
   - If you're using Anaconda, run the following command in your terminal:
     ```
     conda env create -f conda_env.yml
     ```

3. Activate the environment:
   - For Mamba, use the command:
     ```
     micromamba activate bootstop
     ```
   - For Anaconda, use the command:
     ```
     conda activate bootstop
     ```

4. Run the code!
   You should now be ready to run our code. From the root directory of the BootSTOP-more-algorithms repository, simply execute the command:
   ```
   python3 run_<your_CFT_and_optimiser>.py
   ```

## Option 2 - Install with pip (Expert option)

Option 1 is the recommended approach as it handles dependencies more effectively. Option 2 should be used by experienced users who are comfortable managing their environment manually.

If you prefer to use pip, we recommend using pyenv in conjunction with this approach. Please note that users choosing this option should be familiar with installing packages in pyenv environments.

To install with pip, follow the following step:

1. Install the required packages listed in the `requirements.txt` file:
   ```
   pip install -r requirements.txt
   ```

Users who choose this option should be aware that there may be other PyGMO dependencies that are not downloaded. These dependencies must be resolved using the system package manager.

