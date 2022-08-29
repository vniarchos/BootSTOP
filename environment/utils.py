import numpy as np
from csv import writer


def generate_block_list(max_spin, z_kill_list):
    """
    Reads the pregenerated conformal blocks data from csv files into a list.

    Parameters
    ----------
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
        tmp_name = 'block_lattices/6d_blocks_spin' + str(i) + '.csv'
        tmp = np.genfromtxt(tmp_name, delimiter=',')
        # if the kill list is empty append the whole array
        if len(z_kill_list) == 0:
            block_list.append(tmp)
        # otherwise delete the columns which appear in z_kill_list and then append
        else:
            block_list.append(np.delete(tmp, z_kill_list, axis=1))

    print('Done loading pregenerated conformal block data.')
    return block_list


def output_to_file(file_name, output):
    """
    Appends row of output to a file.

    Parameters
    ----------
    file_name : str
        Filename of a writer object.
    output : iterable of strings or numbers
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
