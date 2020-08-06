# /usr/bin/python 
#
# Split an integer across all ranks  
# 
import numpy as np
import paranumpy.paranumpy as pnp  
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

N = 5 

N_loc = pnp.scatter_int ( N )

print ( "On rank ", rank, ' N_loc = ', N_loc )
