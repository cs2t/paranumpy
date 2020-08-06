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

```python
import numpy as np
import paranumpy.paranumpy as pnp
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

N = 5
N_loc = pnp.scatter_int ( N )
print ( "On rank ", rank, ' N_loc = ', N_loc )
```


### Example 2 

```python
import numpy as np
import paranumpy.paranumpy as pnp
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# generate an array of integers
a     = np.asarray ( range (7) ).astype(np.int8)
if (rank == 0 ):
    print (  '\n',"I distribute the following INTEGER array: \n", a , ' \n' )
comm.Barrier()

# distribute the array over all ranks via paranumpy
a_loc = pnp.scatter_1D_array ( a )
print ( "On rank ", rank, ' a_loc = ', a_loc )
comm.Barrier()
```

### Example 3

```python
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
```

### Example 4

```python
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
a = None
Nx = 7
Ny = 7
if (rank == 0):
    a = np.zeros((Nx, Ny),dtype = np.int16)
    for ix in range(Nx):
        for iy in range(Ny):
            a [ix,iy]=(iy) + ( Ny * ix)

if rank == 0 :
    print (  '\n',"Original array: \n", a , ' \n' )

# distribute the array over all ranks via paranumpy
a_loc = pnp.scatter_2D_array ( a )
print ( 'Scattered array on rank ', rank, ':', a_loc, flush = True)
for i in range ( a_loc.shape[0] ):
    a_loc[ i ] =   a_loc[ i ] **2

print ( 'Modified array on rank ', rank, ':', a_loc, flush = True)
a2_gathered = pnp.gather_2D_array (a_loc)
print ( 'Modified (gathered) array on rank ', rank, ':', a2_gathered, flush = True)
a2_allgathered = pnp.allgather_2D_array (a_loc)
print ( 'Modified (allgathered) array on rank ', rank, ':', a2_allgathered)
```

A set of examples which illustrates the usage of paranumpy is given in the folder `./test` 

## Authors

Paranumpy is developed by Fabio Caruso and the Computational Solid-State Theory Laboratory ( https://cs2t.de ).
