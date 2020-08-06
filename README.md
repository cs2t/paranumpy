# Paranumpy  

`paranumpy` provides a set of functions 
to handle numpy arrays in a MPI (mpi4py) parallel environment.

## Installation 

`paranumpy` is still under testing, and it is therefore hosted on the testing platform of the python package index ([PyPI](https://pypi.org)). 
It can be installed with `pip` through the following command:

```bash 
$ pip install -i https://test.pypi.org/simple/  paranumpy
```

Dependencies: `numpy`, `mpi4py`

## Usage 

After successfull installation of the package, 
`paranumpy` can be imported as:  
 
```python
import numpy as np
import paranumpy.paranumpy as pnp
```

Additionally, an MPI instance mush be initialized: 

```python
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
```

A detailed overview of `mpi4py` is available on [the project page](https://mpi4py.readthedocs.io/en/stable/). 
The following examples illustrate some of the basic operations that can be perfomed with  `paranumpy`. 

`paranumpy` implements the following functions: 
- `scatter_int ( N )`: It distributes an integer across all ranks. Input: N (an integer). Output: N_loc (an integer), the value of the distributed integer on each rank. The sum of N_loc on all ranks yields N. 
- `scatter_1D_array   ( a )`: It distributes the entries of a 1D numpy array across all ranks. The array `a` should be defined on rank 0, whereas it can be set to `None` on all other ranks. Input: a (a 1D numpy array). Output: `a_loc` (a 1D numpy array), the distributed array. 
- `gather_1D_array    ( a_loc )`: It gathers the distributed arrays `a_loc` from different ranks, and it returns on rank 0 the global array. Input: `a_loc` (a 1D numpy array). Ouput: a 1D numpy array on rank 0, `None` on other ranks. 
- `allgather_1D_array ( a_loc )`: It gathers the distributed arrays `a_loc` from different ranks, and it returns all ranks the global array. Input: `a_loc` (a 1D numpy array). Ouput: a 1D numpy array. 
-  `scatter_<X>D_array`, `gather_<X>D_array`, `allgather_<X>D_array`: Do the same `scatter_1D_array` etc., for X= 2,3, and 4 2D, 3D, and 4D arrays. The array distribution is done on the first axis of the array. 

The following types for numpy arrays are sypported: 

               np.int8  
               np.int16      
               np.int32      
               np.int64      
               np.uint8      
               np.uint16     
               np.uint32     
               np.uint64     
               np.float32    
               np.float64    
               np.float_     
               np.float128   
               np.complex64  
               np.complex128 
               np.complex_   

## Examples  

These examples are included in the folder examples. 
The parallel execution of the examples can be conducted for instance via: 

```bash
mpirun -np 2 python example01.py 
```



### Example 1 

This first example illustrates how to split the value of an 
integer across the different ranks of a parallel instance. 

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

and it should yield the following output:

```
 On rank  0  N_loc =  3
 On rank  1  N_loc =  2
```

### Example 2 

This examples illustrates the distribution of a 1D numpy array using the 
`scatter_1D_array` function. 

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

Parallel execution of this script on 2 MPI processes should yield:

```
 I distribute the following INTEGER array:
 [0 1 2 3 4 5 6]

On rank  0  a_loc =  [0 1 2 3]
On rank  1  a_loc =  [4 5 6]
```

### Example 3

This examples illustrates how to perform simple parallel operations on a distributed 1D numpy array by using the functions `scatter_1D_array` and `gather_1D_array` or `allgather_1D_array`. 

```python
import numpy as np
import paranumpy.paranumpy as pnp
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

a     = np.asarray ( range (7) ).astype(np.int16)  # generate an array of integers
if rank == 0 :
    print (  '\n',"Original array: \n", a , ' \n' )

a_loc = pnp.scatter_1D_array ( a )  # distribute the array across all ranks
print ( 'Scattered array on rank ', rank, ':', a_loc, flush = True)
for i in range ( a_loc.shape[0] ):
    a_loc[ i ] =   a_loc[ i ] ** 2  # do something in parallel
print ( 'Modified array on rank ', rank, ':', a_loc, flush = True)
a2_gathered = pnp.gather_1D_array (a_loc) # collect the array on rank = 0 
print ( 'Modified (gathered) array on rank ', rank, ':', a2_gathered, flush = True)
a2_allgathered = pnp.allgather_1D_array (a_loc) # collect the array on all ranks
print ( 'Modified (allgathered) array on rank ', rank, ':', a2_allgathered)
```
Parallel execution of this script on 2 MPI processes should yield:

```
 Original array:
 [0 1 2 3 4 5 6]

Scattered array on rank  0 : [0 1 2 3]
Modified array on rank  0 : [0 1 4 9]
Scattered array on rank  1 : [4 5 6]
Modified array on rank  1 : [16 25 36]
Modified (gathered) array on rank  1 : None
Modified (gathered) array on rank  0 : [ 0  1  4  9 16 25 36]
Modified (allgathered) array on rank  0 : [ 0  1  4  9 16 25 36]
Modified (allgathered) array on rank  1 : [ 0  1  4  9 16 25 36]
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
