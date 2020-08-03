
import numpy as np
from mpi4py import MPI
from  distribute_array import * 
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

#print (size, rank) 
print ('Hello from rank = ', rank )

Dtest = 2

# test 1D 
if ( Dtest == 1  ): 
    vec_global = None 
    if (rank == 0 ):
        print (' Testing for D = 1 ')
        vec_global = np.random.random(7)
        for ix in range(vec_global.shape[0]):
            vec_global[ix]= ix
    print('G on ',rank,':', vec_global)
    vec_local = scatter_1D_array_D ( vec_global ) 
    print ('L on ',rank,':', vec_local)
    vec_global = gather_1D_array_D ( vec_local ) 
    print ('G on ',rank,':', vec_global)


if ( Dtest == 2  ): 
    vec_global = None 
    if (rank == 0 ):
        print (' Testing for D = 2 ')
        vec_global = np.random.random((7, 7))
        for ix in range(vec_global.shape[0]):
            for iy in range(vec_global.shape[1]):
                vec_global[ix,iy]=(iy) + ( vec_global.shape[1] * ix)
    
    print('G on ',rank,':', vec_global)
    vec_local = scatter_2D_array_D ( vec_global ) 
    print ('L on ',rank,':', vec_local)
    vec_global = gather_2D_array_D ( vec_local ) 
    print ('G on ',rank,':', vec_global)


if ( Dtest == 3  ): 
    vec_global = None 
    N1 = 4
    N2 = 4 
    N3 = 5
    if (rank == 0 ):
        print (' Testing for D = 3 ')
        vec_global = np.random.random((N1,N2,N3))
        for ix in range(vec_global.shape[0]):
            for iy in range(vec_global.shape[1]):
                for iz in range(vec_global.shape[2]):
                    vec_global[ix,iy,iz]=(iy) + ( vec_global.shape[1] * ix) + ( vec_global.shape[2] * iz)
    
    print('G on ',rank,':', vec_global)
    vec_local = scatter_3D_array_D ( vec_global ) 
    print ('L on ',rank,':', vec_local)
    vec_global = gather_3D_array_D ( vec_local ) 
    print ('G on ',rank,':', vec_global)


if ( Dtest == 4  ): 
    vec_global = None 
    N1 = 2
    N2 = 3 
    N3 = 2
    N4 = 3
    
    if (rank == 0 ):
        print (' Testing for D = 4 ')
        vec_global = np.random.random((N1,N2,N3,N4))
        for ix in range(vec_global.shape[0]):
            for iy in range(vec_global.shape[1]):
                for iz in range(vec_global.shape[2]):
                    for it in range(vec_global.shape[3]):
                        vec_global[ix,iy,iz,it]=(iy) + ( vec_global.shape[1] * ix) + ( vec_global.shape[2] * iz) + ( vec_global.shape[3] * it) 


    print('G on ',rank,':', vec_global)
    vec_local = scatter_4D_array_D ( vec_global ) 
    print ('L on ',rank,':', vec_local)
    vec_global = gather_4D_array_D ( vec_local ) 
    print ('G on ',rank,':', vec_global)
