# Paranumpy  

`paranumpy` is conceived as a tool to facilitate the parallelization of numpy arrays in python. 
It consists of a set of functions to handle numpy arrays in a MPI (mpi4py) parallel environment.

# Installation 

`paranumpy` is still under testing. It is hosted on the python package index ([PyPI](https://pypi.org)) and
it can be installed with `pip`:

```bash 
$ pip install paranumpy
```

Dependencies: `numpy`, `mpi4py`

# Usage 

After successful installation `paranumpy` can be imported as an ordinary module:  
 
```python
import numpy as np
import paranumpy.paranumpy as pnp
```

An MPI instance must be initialized: 

```python
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
```

A detailed overview of `mpi4py` is available on [the project page](https://mpi4py.readthedocs.io/en/stable/). 

`paranumpy` implements the following functions: 
- `scatter_int ( N )`: It distributes an integer across all ranks. *Arguments*: N (an integer). It returns an integer N_loc, which is the value of the distributed integer on each rank. The sum of N_loc on all ranks yields N. 
- `scatter_1D_array   ( a )`: It distributes the entries of a 1D numpy array across all ranks. The array `a` should be defined on rank 0, whereas it can be set to `None` on all other ranks. *Arguments*: a (a 1D numpy array). The function returns a 1D numpy array `a_loc`, which is the distributed 1D array. 
- `gather_1D_array    ( a_loc )`: It gathers the distributed array `a_loc` from different ranks, and it returns on rank 0 the global array. *Arguments*: `a_loc` (a 1D numpy array). Output: a 1D numpy array on rank 0, `None` on other ranks. 
- `allgather_1D_array ( a_loc )`: It gathers the distributed array `a_loc` from different ranks, and it returns all ranks the global array. *Arguments*: `a_loc` (a 1D numpy array). Output: a 1D numpy array. 
-  `scatter_<X>D_array`, `gather_<X>D_array`, `allgather_<X>D_array`: Do the same `scatter_1D_array` etc., for X= 2,3, and 4 2D, 3D, and 4D arrays. The array distribution is done on the first axis of the array. 

The following types for numpy arrays are supported: 

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

# Examples  

The following examples illustrate some of the basic operations that can be performed with  `paranumpy`. 
These examples are included in the folder examples. 
The parallel execution of the examples can be conducted for instance via: 

```bash
mpirun -np 2 python example01.py 
```


## Example 1 

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

## Example 2 

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

## Example 3

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
print ( 'Scattered array on rank ', rank, ':', a_loc)

for i in range ( a_loc.shape[0] ):
    a_loc[ i ] =   a_loc[ i ] ** 2  # do something in parallel
print ( 'Modified array on rank ', rank, ':', a_loc)

a2_gathered = pnp.gather_1D_array (a_loc) # collect the array on rank = 0 
print ( 'Modified (gathered) array on rank ', rank, ':', a2_gathered)

a2_allgathered = pnp.allgather_1D_array (a_loc) # collect the array on all ranks
print ( 'Modified (allgathered) array on rank ', rank, ':', a2_allgathered)
```
Parallel execution of this script on 2 MPI processes should yield:

```
 Original array:
 [0 1 2 3 4 5 6]

Scattered array on rank  0 : [0 1 2 3]
Scattered array on rank  1 : [4 5 6]
Modified array on rank  0 : [0 1 4 9]
Modified array on rank  1 : [16 25 36]
Modified (gathered) array on rank  0 : [ 0  1  4  9 16 25 36]
Modified (gathered) array on rank  1 : None
Modified (allgathered) array on rank  0 : [ 0  1  4  9 16 25 36]
Modified (allgathered) array on rank  1 : [ 0  1  4  9 16 25 36]
```


## Example 4

This examples essentially repeat the same steps of example 3, for a 2D array. 

```python
import numpy as np
import paranumpy.paranumpy as pnp
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# generate a 2D array of integers on rank 0 
a = None
Nx = 3
Ny = 4
if (rank == 0):
    a = np.zeros((Nx, Ny),dtype = np.int16)
    for ix in range(Nx):
        for iy in range(Ny):
            a [ix,iy]=(iy) + ( Ny * ix)

if rank == 0 :
    print (  '\n',"Original array: \n", a , ' \n' )

# distribute the array over all ranks via paranumpy
a_loc = pnp.scatter_2D_array ( a )
print ( 'Scattered array on rank ', rank, ':\n', a_loc)
for i in range ( a_loc.shape[0] ):
    a_loc[ i ] =   a_loc[ i ] **2

print ( 'Modified array on rank ', rank, ':\n', a_loc)
a2_gathered = pnp.gather_2D_array (a_loc)

print ( 'Modified (gathered) array on rank ', rank, ':\n', a2_gathered)
a2_allgathered = pnp.allgather_2D_array (a_loc)
print ( 'Modified (allgathered) array on rank ', rank, ':\n', a2_allgathered)
```

Parallel execution of this script on 2 MPI processes should yield:
```
 Original array:
 [[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]

Scattered array on rank  0 :
 [[0 1 2 3]
 [4 5 6 7]]
Scattered array on rank  1 :
 [[ 8  9 10 11]]
Modified array on rank  0 :
 [[ 0  1  4  9]
 [16 25 36 49]]
Modified array on rank  1 :
 [[ 64  81 100 121]]
Modified (gathered) array on rank  0 :
 [[  0   1   4   9]
 [ 16  25  36  49]
 [ 64  81 100 121]]
Modified (gathered) array on rank  1 :
 None
Modified (allgathered) array on rank  1 :
 [  0   1   4   9]
 [ 16  25  36  49]
 [ 64  81 100 121]]
Modified (allgathered) array on rank  0 :
 [[  0   1   4   9]
 [ 16  25  36  49]
 [ 64  81 100 121]]
```

# Authors

Paranumpy is developed by Fabio Caruso and the Computational Solid-State Theory Laboratory ([https://cs2t.de](https://cs2t.de)).
