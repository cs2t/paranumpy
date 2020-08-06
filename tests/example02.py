# /usr/bin/python 
#
# Distribute a 1D array across ranks.
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
a     = np.asarray ( range (7) ).astype(np.int8)  
if (rank == 0 ):
    print (  '\n',"I distribute the following INTEGER array: \n", a , ' \n' )
comm.Barrier()

# distribute the array over all ranks via paranumpy 
a_loc = pnp.scatter_1D_array ( a )
print ( "On rank ", rank, ' a_loc = ', a_loc )
comm.Barrier()


##############
# FLOAT 
#
# generate an array of random floats 
np.random.seed(12345)
a     = np.asarray ( np.random.random(7) ).astype(np.float64) 
if (rank == 0 ):
    print (  '\n',"I distribute the following FLOAT array: \n", a , ' \n' )
comm.Barrier()

# distribute the array over all ranks via paranumpy 
a_loc = pnp.scatter_1D_array ( a )
print ( "On rank ", rank, ' a_loc = ', a_loc )
comm.Barrier()


##############
# COMPLEX 
#
# generate an array of random complex numbers
np.random.seed(12345)
a     = np.asarray ( np.random.random(7) ).astype(np.complex64) 
np.random.seed(23456)
a     += 1J * np.asarray ( np.random.random(7) )
if (rank == 0 ):
    print (  '\n',"I distribute the following COMPLEX array: \n", a , ' \n' )
comm.Barrier()

# distribute the array over all ranks via paranumpy 
a_loc = pnp.scatter_1D_array ( a )
print ( "On rank ", rank, ' a_loc = ', a_loc )
comm.Barrier()

