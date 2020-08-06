# Paranumpy  

Paranumpy provides a set of functions 
to handle numpy arrays in a MPI (mpi4py) parallel environment.

## Installation 

Paranumpy is still under testing, and it is therefore hosted on the testing platform of PyPI. 
Paranumpy can be installed via the following command:

> pip install -i https://test.pypi.org/simple/  paranumpy 

Dependencies: `numpy`, `mpi4py`

## Usage 

### Example 1 


>import numpy as np
>import paranumpy.paranumpy as pnp
>from mpi4py import MPI
>
>comm = MPI.COMM_WORLD
>size = comm.Get_size()
>rank = comm.Get_rank()
>
>N = 5
>N_loc = pnp.scatter_int ( N )
>print ( "On rank ", rank, ' N_loc = ', N_loc )


### Example 2 

>import numpy as np
>import paranumpy.paranumpy as pnp
>from mpi4py import MPI
>
>comm = MPI.COMM_WORLD
>size = comm.Get_size()
>rank = comm.Get_rank()
>
># generate an array of integers
>a     = np.asarray ( range (7) ).astype(np.int8)
>if (rank == 0 ):
>    print (  '\n',"I distribute the following INTEGER array: \n", a , ' \n' )
>comm.Barrier()
>
># distribute the array over all ranks via paranumpy
>a_loc = pnp.scatter_1D_array ( a )
>print ( "On rank ", rank, ' a_loc = ', a_loc )
>comm.Barrier()


A set of examples which illustrates the usage of paranumpy is given in the folder `./test` 

## Authors

Paranumpy is developed by Fabio Caruso and the Computational Solid-State Theory Laboratory ( https://cs2t.de ).
