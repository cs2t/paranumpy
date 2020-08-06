# /usr/bin/python 
#
# (1) Distribute a 1D array across ranks
# (2) Perform some simple operation in parallel
# (3) Gather the array on rank = 0 
# 
import numpy as np
import paranumpy.paranumpy as pnp  
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

##############
# INTEGERS 
#
# generate an array of integers
a     = np.asarray ( range (7) ).astype(np.int16)  
if rank == 0 : 
    print (  '\n',"Original array: \n", a , ' \n' )
# distribute the array over all ranks via paranumpy 
a_loc = pnp.scatter_1D_array ( a )
print ( 'Scattered array on rank ', rank, ':', a_loc, flush = True)
for i in range ( a_loc.shape[0] ):
    a_loc[ i ] =   a_loc[ i ] ** 2
print ( 'Modified array on rank ', rank, ':', a_loc, flush = True)
a2_gathered = pnp.gather_1D_array (a_loc)
print ( 'Modified (gathered) array on rank ', rank, ':', a2_gathered, flush = True)
a2_allgathered = pnp.allgather_1D_array (a_loc)
print ( 'Modified (allgathered) array on rank ', rank, ':', a2_allgathered)
