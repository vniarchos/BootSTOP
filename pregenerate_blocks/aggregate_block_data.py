import numpy as np
import os
import sys

spin = str(sys.argv[1])
num_of_files = 30
# num_of_files should be the same as the number of array jobs sent to Apocrita
# and equates to the final size of the conformal weight range i.e. [5.8 + spin, 5.8 + spin + 30)

cur_dir = os.getcwd()
os.chdir(cur_dir + '/Mca/')

for i in range(1, num_of_files + 1):
    filename = '/6d_blocks_spin' + str(spin) + '_' + str(i) + '.csv'
    # read in csv
    temp = np.genfromtxt(os.getcwd() + filename, delimiter=',')
    print('shape ' + str(i) + ' = ' + str(temp.shape))
    if i == 1:
        aggregate = temp
    else:
        aggregate = np.concatenate((aggregate, temp))

print('aggregate shape = ' + str(aggregate.shape))
# reshape the aggregate array here, note I'm using 180 z-points and 0.0005 delta spacing
agg_rs = aggregate.reshape(num_of_files * 2000, 180)
# check reshape worked as expected
print('aggregate reshape = ' + str(agg_rs.shape))

# save the aggregate array to a csv file
np.savetxt(cur_dir + '/6d_blocks_spin' + str(spin) + '.csv', agg_rs, fmt='%.15f', delimiter=',')
