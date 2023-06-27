import numpy as np
from csv import writer


def generate_block_list(block_type, max_spin, z_kill_list):
    """
    Reads the pregenerated conformal blocks data from csv files into a list.

    Parameters
    ----------
    block_type : {'6d_blocks_spin', '2d_blocks_a_spin', '2d_blocks_b_spin'}
        What type of block to be loaded
    max_spin : int
        The upper bound of spin for loading conformal blocks.
    z_kill_list : list
        A list of z-point positions to remove.
    Returns
    -------
    block_list : list
        A list of ndarrays containing pregenerated conformal block data.
    """
    # since this takes a long time give the user some feedback
    print('Loading pregenerated conformal block data.')
    block_list = []
    for i in range(0, max_spin + 2, 2):
        tmp_name = 'block_lattices/' + block_type + str(i) + '.csv'
        tmp = np.genfromtxt(tmp_name, delimiter=',')
        # if the kill list is empty append the whole array
        if len(z_kill_list) == 0:
            block_list.append(tmp)
        # otherwise delete the columns which appear in z_kill_list and then append
        else:
            block_list.append(np.delete(tmp, z_kill_list, axis=1))

    print('Done loading pregenerated conformal block data.')
    return block_list


def generate_block_list_npy(block_directory, suffixes, index_kill_list):
    """
    Reads the pregenerated conformal blocks data from npy files into a list.

    Parameters
    ----------
    block_directory : str
        A string containing the name of a subdirectory of `block_lattices` and filename of type of block to be loaded.
    suffixes : list
        A list of strings or numbers to iterate over.
    index_kill_list : list
        A list of positions to remove from each block so that we can take a subset of total pre-generated data.
    Returns
    -------
    block_list : list
        A list of ndarrays containing pregenerated conformal block data.
    """
    # since this can take a long time give the user some feedback
    print('Loading pregenerated conformal block data.')
    block_list = []
    for suffix in suffixes:
        block = np.load('block_lattices/' + block_directory + str(suffix) + '.npy')
        # if the kill list is not empty then delete a part of the block array
        if len(index_kill_list) != 0:
            block = np.delete(block, index_kill_list, axis=1)  # delete does not happen in-place so need to update block
        block_list.append(block)

    print('Done loading pregenerated conformal block data.')
    return block_list


def generate_block_list_csv(block_type, suffixes, kill_list):
    """
    Reads the pregenerated conformal blocks data from csv files into a list.

    Parameters
    ----------
    block_type : {'6d_blocks_spin', '2d_blocks_a_spin', '2d_blocks_b_spin'}
        What type of block to be loaded
    suffixes : list
        A list of strings or numbers to iterate over.
    kill_list : list
        A list of z-point positions to remove.
    Returns
    -------
    block_list : list
        A list of ndarrays containing pregenerated conformal block data.
    """
    block_list = []
    for suffix in suffixes:
        block = np.genfromtxt('block_lattices/' + block_type + str(suffix) + '.csv', delimiter=',')
        # if the kill list is not empty then delete a part of the block array
        if len(kill_list) != 0:
            block = np.delete(block, kill_list, axis=1)  # delete does not happen in-place so need to update block
        block_list.append(block)

    return block_list


def output_to_file(file_name, output):
    """
    Appends row of output to a file.

    Parameters
    ----------
    file_name : str
        Filename of a writer object.
    output : ndarray
        The parameter passed to writer.writerow.

    """
    with open(file_name, 'a', newline='') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(output)
        f_object.close()


def output_to_console(output):
    """
    Print to the console.

    Parameters
    ----------
    output : str
        String printed to the console.
    """
    print(output)
