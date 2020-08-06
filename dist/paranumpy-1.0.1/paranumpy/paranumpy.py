
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

mpi_types  = { np.int8       : MPI.INT8_T , 
               np.int16      : MPI.INT16_T , 
               np.int32      : MPI.INT32_T , 
               np.int64      : MPI.INT64_T , 
               np.uint8      : MPI.UINT8_T , 
               np.uint16     : MPI.UINT16_T , 
               np.uint32     : MPI.UINT32_T , 
               np.uint64     : MPI.UINT64_T ,
               np.float32    : MPI.FLOAT ,
               np.float64    : MPI.DOUBLE , 
               np.float_     : MPI.DOUBLE , 
               np.float128   : MPI.LONG_DOUBLE, 
               np.complex64  : MPI.COMPLEX,
               np.complex128 : MPI.DOUBLE_COMPLEX,
               np.complex_   : MPI.DOUBLE_COMPLEX
               }


#-----------------------------------------------
def scatter_int (N):
    a0    = np.zeros(N,dtype=np.int8)
    split = np.array_split(a0,size)
    
    split_sizes = []
    for i in range(0,len(split),1):
        split_sizes = np.append(split_sizes, len(split[i]))
    split_sizes = np.asarray(split_sizes,dtype=np.int32) 
    N_loc = comm.scatter ( split_sizes, root = 0)
    return N_loc


#-----------------------------------------------
def scatter_1D_array (vec_global):
    dtype = None 
    if rank == 0:
        dtype = type( vec_global [0] )
        vec_global = np.ascontiguousarray(vec_global, dtype=dtype)
        N1 = vec_global.shape[0]
        split = np.array_split(vec_global,size,axis = 0) #Split input array by the number of available cores
        split_sizes = []
        for i in range(0,len(split),1):
            split_sizes = np.append(split_sizes, len(split[i]))
        split_sizes_input = split_sizes
        displacements_input = np.insert(np.cumsum(split_sizes_input),0,0)[0:-1]
    else:
    #Create variables on other cores
        split_sizes_input = None
        displacements_input = None
        split = None
        vec_global = None

    dtype = comm.bcast (dtype,root=0)
    split = comm.bcast(split, root=0) #Broadcast split array to other cores
    vec_local = np.zeros(np.shape(split[rank]),dtype=dtype) #Create array to receive subset of data on each core, where rank specifies the core
    if dtype in mpi_types:
        comm.Scatterv([vec_global,split_sizes_input, displacements_input,mpi_types[dtype]],vec_local,root=0)
    else : 
        print( " ERROR:", dtype ," is still an unsupported datatype ")
    return vec_local

#-----------------------------------------------
def gather_1D_array ( vec_local ):

        # get the data type from the array on rank 0
        dtype = None 
        if  rank == 0:
           dtype = type (vec_local[0])
        dtype = comm.bcast (dtype,root=0)

        vec_local =  np.ascontiguousarray(vec_local, dtype=dtype)
        N1_loc = vec_local.shape[0]
        N1 =  comm.allreduce(N1_loc,op=MPI.SUM)   # recover full size along first dimension

        if rank == 0:
            vec_global = np.zeros([N1],dtype=dtype)             #Create output array of same size
        else:
            vec_global = None

        split_size_loc = vec_local.shape[0]
        split_size = np.asarray ( comm.gather (split_size_loc, root=0))

        if rank == 0 :
            split_sizes_output = split_size
            displacements_output = np.insert(np.cumsum(split_sizes_output),0,0)[0:-1]
        else :
            split_sizes_output = None
            displacements_output = None

        split_sizes_output = comm.bcast(split_sizes_output, root = 0)
        displacements_output = comm.bcast(displacements_output, root = 0)

        comm.Barrier()
        if dtype in mpi_types:
            comm.Gatherv(vec_local,[vec_global,split_sizes_output,displacements_output,mpi_types[dtype]], root=0) #Gather output data together
        else : 
            print( " ERROR:", dtype ," is still an unsupported datatype ")
        return vec_global


#-----------------------------------------------
def allgather_1D_array ( vec_local ):

        # get the data type from the array on rank 0
        dtype = None 
        if  rank == 0:
           dtype = type (vec_local[0])
        dtype = comm.bcast (dtype,root=0)

        vec_local =  np.ascontiguousarray(vec_local, dtype=dtype)
        N1_loc = vec_local.shape[0]
        N1 =  comm.allreduce(N1_loc,op=MPI.SUM)   # recover full size along first dimension

        vec_global = np.zeros([N1],dtype=dtype)             #Create output array of same size

        split_size_loc = vec_local.shape[0]
        split_size = np.asarray ( comm.gather (split_size_loc, root=0))

        split_sizes_output = split_size
        displacements_output = np.insert(np.cumsum(split_sizes_output),0,0)[0:-1]

        split_sizes_output = comm.bcast(split_sizes_output, root = 0)
        displacements_output = comm.bcast(displacements_output, root = 0)

        comm.Barrier()
        if dtype in mpi_types:
            comm.Allgatherv(vec_local,[vec_global,split_sizes_output,displacements_output,mpi_types[dtype]]) #Gather output data together
        else : 
            print( " ERROR:", dtype ," is still an unsupported datatype ")
        return vec_global


# scatters a 2D double array over all ranks

#-----------------------------------------------
def scatter_2D_array ( vec_global ):


    # get the data type from the array on rank 0
    dtype = None 
    if  rank == 0:
       dtype = type (vec_global [0,0] )
    dtype = comm.bcast (dtype,root=0)

    if rank == 0:
        vec_global = np.ascontiguousarray(vec_global, dtype=dtype)
        N1 = vec_global.shape[0]
        N2 = vec_global.shape[1]

        split = np.array_split(vec_global,size,axis = 0) #Split input array by the number of available cores
        split_sizes = []
        for i in range(0,len(split),1):
            split_sizes = np.append(split_sizes, len(split[i]))

        split_sizes_input = split_sizes * N2
        displacements_input = np.insert(np.cumsum(split_sizes_input),0,0)[0:-1]

    else:
    #Create variables on other cores
        split_sizes_input = None
        displacements_input = None
        split = None
        vec_global = None

    split = comm.bcast(split, root=0) #Broadcast split array to other cores
    vec_local = np.zeros(np.shape(split[rank]),dtype=dtype) #Create array to receive subset of data on each core, where rank specifies the core

    if dtype in mpi_types:
        comm.Scatterv([vec_global,split_sizes_input, displacements_input,mpi_types[dtype]],vec_local,root=0)
    else : 
        print( " ERROR:", dtype ," is still an unsupported datatype ")

    return vec_local



#-----------------------------------------------
def gather_2D_array ( vec_local  ):

        # get the data type from the array on rank 0
        dtype = None 
        if  rank == 0:
           dtype = type (vec_local [0,0] )
        dtype = comm.bcast (dtype,root=0)

        vec_local =  np.ascontiguousarray(vec_local, dtype=dtype)
        N1_loc = vec_local.shape[0]
        N2     = vec_local.shape[1]

        N1 =  comm.allreduce(N1_loc,op=MPI.SUM)   # recover full size along first dimension

        if rank == 0:
            vec_global = np.zeros([N1,N2],dtype=dtype)             #Create output array of same size
        else:
            vec_global = None

        split_size_loc = vec_local.shape[0]
        split_size = np.asarray ( comm.gather (split_size_loc, root=0))

        if rank == 0 :
            split_sizes_output = split_size * N2
            displacements_output = np.insert(np.cumsum(split_sizes_output),0,0)[0:-1]
        else :
            split_sizes_output = None
            displacements_output = None
        split_sizes_output = comm.bcast(split_sizes_output, root = 0)
        displacements_output = comm.bcast(displacements_output, root = 0)

        comm.Barrier()

        if dtype in mpi_types:
            comm.Gatherv(vec_local,[vec_global,split_sizes_output,displacements_output,mpi_types[dtype]], root=0) #Gather output data together
        else : 
            print( " ERROR:", dtype ," is still an unsupported datatype ")

        return vec_global

#-----------------------------------------------
def allgather_2D_array ( vec_local ):

        # get the data type from the array on rank 0
        dtype = None 
        if  rank == 0:
           dtype = type (vec_local [0,0] )
        dtype = comm.bcast (dtype,root=0)

        vec_local =  np.ascontiguousarray(vec_local, dtype=dtype)
        N1_loc = vec_local.shape[0]
        N2     = vec_local.shape[1]

        N1 =  comm.allreduce(N1_loc,op=MPI.SUM)   # recover full size along first dimension

        vec_global = np.zeros([N1,N2],dtype=dtype)             #Create output array of same size

        split_size_loc = vec_local.shape[0]
        split_size = np.asarray ( comm.gather (split_size_loc, root=0))

        if rank == 0 :
            split_sizes_output = split_size * N2
            displacements_output = np.insert(np.cumsum(split_sizes_output),0,0)[0:-1]
        else :
            split_sizes_output = None
            displacements_output = None
        split_sizes_output = comm.bcast(split_sizes_output, root = 0)
        displacements_output = comm.bcast(displacements_output, root = 0)

        comm.Barrier()

        if dtype in mpi_types:
            comm.Allgatherv(vec_local,[vec_global,split_sizes_output,displacements_output,mpi_types[dtype]]) #Gather output data together
        else : 
            print( " ERROR:", dtype ," is still an unsupported datatype ")
        return vec_global


#-----------------------------------------------
def scatter_3D_array ( vec_global ):


    # get the data type from the array on rank 0
    dtype = None
    if  rank == 0:
       dtype = type (vec_global [0,0,0] )
    dtype = comm.bcast (dtype,root=0)

    if rank == 0:
        vec_global = np.ascontiguousarray(vec_global, dtype=dtype)
        N1 = vec_global.shape[0]
        N2 = vec_global.shape[1]
        N3 = vec_global.shape[2]

        split = np.array_split(vec_global,size,axis = 0) #Split input array by the number of available cores
        split_sizes = []
        for i in range(0,len(split),1):
            split_sizes = np.append(split_sizes, len(split[i]))

        split_sizes_input = split_sizes * N2 * N3
        displacements_input = np.insert(np.cumsum(split_sizes_input),0,0)[0:-1]

    else:
    #Create variables on other cores
        split_sizes_input = None
        displacements_input = None
        split = None
        vec_global = None

    split = comm.bcast(split, root=0) #Broadcast split array to other cores
    vec_local = np.zeros(np.shape(split[rank]), dtype = dtype) #Create array to receive subset of data on each core, where rank specifies the core

    if dtype in mpi_types:
        comm.Scatterv([vec_global,split_sizes_input, displacements_input,mpi_types[dtype]],vec_local,root=0)
    else :
        print( " ERROR:", dtype ," is still an unsupported datatype ")

    return vec_local


#-----------------------------------------------
def gather_3D_array ( vec_local ):

        # get the data type from the array on rank 0
        dtype = None
        if  rank == 0:
           dtype = type (vec_local [0,0,0] )
        dtype = comm.bcast (dtype,root=0)

        vec_local =  np.ascontiguousarray(vec_local, dtype=dtype)
        N1_loc = vec_local.shape[0]
        N2     = vec_local.shape[1]
        N3     = vec_local.shape[2]

        N1 =  comm.allreduce(N1_loc,op=MPI.SUM)   # recover full size along first dimension

        if rank == 0:
            vec_global = np.zeros([N1,N2,N3],dtype = dtype)             #Create output array of same size
        else:
            vec_global = None

        split_size_loc = vec_local.shape[0]
        split_size = np.asarray ( comm.gather (split_size_loc, root=0))

        if rank == 0 :
            split_sizes_output = split_size * N2 * N3
            displacements_output = np.insert(np.cumsum(split_sizes_output),0,0)[0:-1]
        else :
            split_sizes_output = None
            displacements_output = None
        split_sizes_output = comm.bcast(split_sizes_output, root = 0)
        displacements_output = comm.bcast(displacements_output, root = 0)

        comm.Barrier()

        if dtype in mpi_types:
            comm.Gatherv(vec_local,[vec_global,split_sizes_output,displacements_output,mpi_types[dtype]], root=0) #Gather output data together
        else :
            print( " ERROR:", dtype ," is still an unsupported datatype ")

        return vec_global


#-----------------------------------------------
def allgather_3D_array ( vec_local  ):

        # get the data type from the array on rank 0
        dtype = None
        if  rank == 0:
           dtype = type (vec_local [0,0,0] )
        dtype = comm.bcast (dtype,root=0)

        vec_local =  np.ascontiguousarray(vec_local, dtype=dtype)
        N1_loc = vec_local.shape[0]
        N2     = vec_local.shape[1]
        N3     = vec_local.shape[2]

        N1 =  comm.allreduce(N1_loc,op=MPI.SUM)   # recover full size along first dimension

        vec_global = np.zeros([N1,N2,N3],dtype=dtype)             #Create output array of same size

        split_size_loc = vec_local.shape[0]
        split_size = np.asarray ( comm.gather (split_size_loc, root=0))

        if rank == 0 :
            split_sizes_output = split_size * N2 * N3
            displacements_output = np.insert(np.cumsum(split_sizes_output),0,0)[0:-1]
        else :
            split_sizes_output = None
            displacements_output = None
        split_sizes_output = comm.bcast(split_sizes_output, root = 0)
        displacements_output = comm.bcast(displacements_output, root = 0)

        comm.Barrier()

        if dtype in mpi_types:
            comm.Allgatherv(vec_local,[vec_global,split_sizes_output,displacements_output,mpi_types[dtype]]) #Gather output data together
        else :
            print( " ERROR:", dtype ," is still an unsupported datatype ")
        return vec_global



#-----------------------------------------------
def scatter_4D_array ( vec_global ):

    # get the data type from the array on rank 0
    dtype = None
    if  rank == 0:
       dtype = type (vec_global [0,0,0,0] )
    dtype = comm.bcast (dtype,root=0)

    if rank == 0:
        vec_global = np.ascontiguousarray(vec_global, dtype=dtype)
        N1 = vec_global.shape[0]
        N2 = vec_global.shape[1]
        N3 = vec_global.shape[2]
        N4 = vec_global.shape[3]

        split = np.array_split(vec_global,size,axis = 0) #Split input array by the number of available cores
        split_sizes = []
        for i in range(0,len(split),1):
            split_sizes = np.append(split_sizes, int( len(split[i]) ) )

        split_sizes_input = split_sizes * N2 * N3 * N4
        displacements_input = np.insert(np.cumsum(split_sizes_input),0,0)[0:-1]

    else:
    #Create variables on other cores
        split_sizes_input = None
        displacements_input = None
        split = None
        vec_global = None

    split = comm.bcast(split, root=0) #Broadcast split array to other cores
    vec_local = np.zeros(np.shape(split[rank]),dtype=dtype) #Create array to receive subset of data on each core, where rank specifies the core


    if dtype in mpi_types:
        comm.Scatterv([vec_global,split_sizes_input, displacements_input,mpi_types[dtype]],vec_local,root=0)
    else :
        print( " ERROR:", dtype ," is still an unsupported datatype ")

    return vec_local


#-----------------------------------------------
def gather_4D_array ( vec_local ):

        # get the data type from the array on rank 0
        dtype = None
        if  rank == 0:
           dtype = type (vec_local [0,0,0,0] )
        dtype = comm.bcast (dtype,root=0)

        vec_local =  np.ascontiguousarray(vec_local, dtype=dtype)
        N1_loc = vec_local.shape[0]
        N2     = vec_local.shape[1]
        N3     = vec_local.shape[2]
        N4     = vec_local.shape[3]

        N1 =  comm.allreduce(N1_loc,op=MPI.SUM)   # recover full size along first dimension

        if rank == 0:
            vec_global = np.zeros([N1,N2,N3,N4],dtype=dtype)             #Create output array of same size
        else:
            vec_global = None

        split_size_loc = vec_local.shape[0]
        split_size = np.asarray ( comm.gather (split_size_loc, root=0))

        if rank == 0 :
            split_sizes_output = split_size * N2 * N3 * N4
            displacements_output = np.insert(np.cumsum(split_sizes_output),0,0)[0:-1]
        else :
            split_sizes_output = None
            displacements_output = None
        split_sizes_output = comm.bcast(split_sizes_output, root = 0)
        displacements_output = comm.bcast(displacements_output, root = 0)

        comm.Barrier()

        if dtype in mpi_types:
            comm.Gatherv(vec_local,[vec_global,split_sizes_output,displacements_output,mpi_types[dtype]], root=0) #Gather output data together
        else :
            print( " ERROR:", dtype ," is still an unsupported datatype ")

        return vec_global


#-----------------------------------------------
def allgather_4D_array ( vec_local ):

        # get the data type from the array on rank 0
        dtype = None
        if  rank == 0:
           dtype = type (vec_local [0,0,0,0] )
        dtype = comm.bcast (dtype,root=0)

        vec_local = np.ascontiguousarray(vec_local, dtype=dtype)
        N1_loc = vec_local.shape[0]
        N2     = vec_local.shape[1]
        N3     = vec_local.shape[2]
        N4     = vec_local.shape[3]

        N1 =  comm.allreduce(N1_loc,op=MPI.SUM)   # recover full size along first dimension

        vec_global = np.zeros([N1,N2,N3,N4], dtype=dtype)             #Create output array of same size

        split_size_loc = vec_local.shape[0]
        split_size = np.asarray ( comm.gather (split_size_loc, root=0))

        if rank == 0 :
            split_sizes_output = split_size * N2 * N3 * N4
            displacements_output = np.insert(np.cumsum(split_sizes_output),0,0)[0:-1]
        else :
            split_sizes_output = None
            displacements_output = None
        split_sizes_output = comm.bcast(split_sizes_output, root = 0)
        displacements_output = comm.bcast(displacements_output, root = 0)

        comm.Barrier()

        if dtype in mpi_types:
            comm.Allgatherv(vec_local,[vec_global,split_sizes_output,displacements_output,mpi_types[dtype]]) #Gather output data together
        else :
            print( " ERROR:", dtype ," is still an unsupported datatype ")

        return vec_global
