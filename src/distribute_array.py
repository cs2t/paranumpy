
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def scatter_int (N):
    a0    = np.zeros(N,dtype=np.int8)
    split = np.array_split(a0,size)
    
    split_sizes = []
    for i in range(0,len(split),1):
        split_sizes = np.append(split_sizes, len(split[i]))
    split_sizes = np.asarray(split_sizes,dtype=np.int32) 
    N_loc = comm.scatter ( split_sizes, root = 0)
    return N_loc


def scatter_1D_array (vec_global,dtype = np.float64):
    if rank == 0:
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

    split = comm.bcast(split, root=0) #Broadcast split array to other cores
    vec_local = np.zeros(np.shape(split[rank]),dtype=dtype) #Create array to receive subset of data on each core, where rank specifies the core
    if   dtype == np.float64 :
        comm.Scatterv([vec_global,split_sizes_input, displacements_input,MPI.DOUBLE],vec_local,root=0)
    elif dtype == np.int32 :
        comm.Scatterv([vec_global,split_sizes_input, displacements_input,MPI.INT],vec_local,root=0)
    return vec_local

def gather_1D_array ( vec_local,dtype = np.float64 ):

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
#             print("Input data split into vectors of sizes %s" %split_sizes_output )
#             print("Input data split with displacements of %s" %displacements_output)
        else :
            split_sizes_output = None
            displacements_output = None

        split_sizes_output = comm.bcast(split_sizes_output, root = 0)
        displacements_output = comm.bcast(displacements_output, root = 0)

        comm.Barrier()
        if   dtype == np.int32 :
            comm.Gatherv(vec_local,[vec_global,split_sizes_output,displacements_output,MPI.INT], root=0) #Gather output data together
        if   dtype == np.float64 :
            comm.Gatherv(vec_local,[vec_global,split_sizes_output,displacements_output,MPI.DOUBLE], root=0) #Gather output data together
        return vec_global


def allgather_1D_array ( vec_local, dtype = np.float64 ):

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
        if   dtype == np.int32 :
            comm.Allgatherv(vec_local,[vec_global,split_sizes_output,displacements_output,MPI.INT]) #Gather output data together
        if   dtype == np.float64 :
            comm.Allgatherv(vec_local,[vec_global,split_sizes_output,displacements_output,MPI.DOUBLE]) #Gather output data together
        return vec_global


# scatters a 2D double array over all ranks

def scatter_2D_array ( vec_global, dtype=np.float64 ):

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
    if   dtype == np.float64:
        comm.Scatterv([vec_global,split_sizes_input, displacements_input,MPI.DOUBLE],vec_local,root=0)
    elif dtype == np.int32:
        comm.Scatterv([vec_global,split_sizes_input, displacements_input,MPI.INT],vec_local,root=0)

    return vec_local



def gather_2D_array ( vec_local , dtype=np.float64):

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
        if   dtype == np.float64:
            comm.Gatherv(vec_local,[vec_global,split_sizes_output,displacements_output,MPI.DOUBLE], root=0) #Gather output data together
        if   dtype == np.int32:
            comm.Gatherv(vec_local,[vec_global,split_sizes_output,displacements_output,MPI.INT], root=0) #Gather output data together
        return vec_global

def allgather_2D_array ( vec_local , dtype=np.float64):

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
        if   dtype == np.float64:
            comm.Allgatherv(vec_local,[vec_global,split_sizes_output,displacements_output,MPI.DOUBLE]) #Gather output data together
        if   dtype == np.int32:
            comm.Allgatherv(vec_local,[vec_global,split_sizes_output,displacements_output,MPI.INT]) #Gather output data together
        return vec_global


def scatter_3D_array ( vec_global, dtype = np.float64 ):

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
    if dtype == np.float64 :
        comm.Scatterv([vec_global,split_sizes_input, displacements_input,MPI.DOUBLE],vec_local,root=0)
    if dtype == np.int32 :
        comm.Scatterv([vec_global,split_sizes_input, displacements_input,MPI.INT],vec_local,root=0)
    return vec_local


def gather_3D_array ( vec_local, dtype = np.float64  ):
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
        if dtype == np.float64 :
            comm.Gatherv(vec_local,[vec_global,split_sizes_output,displacements_output,MPI.DOUBLE], root=0) #Gather output data together
        if dtype == np.int32 :
            comm.Gatherv(vec_local,[vec_global,split_sizes_output,displacements_output,MPI.INT], root=0) #Gather output data together

        return vec_global


def allgather_3D_array ( vec_local , dtype = np.float64  ):

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
        if dtype == np.float64 :
            comm.Allgatherv(vec_local,[vec_global,split_sizes_output,displacements_output,MPI.DOUBLE]) #Gather output data together
        if dtype == np.int32 :
            comm.Allgatherv(vec_local,[vec_global,split_sizes_output,displacements_output,MPI.INT]) #Gather output data together
        return vec_global



def scatter_4D_array ( vec_global, dtype = np.float64 ):
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
            #print ( split_sizes )

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
    if dtype == np.float64 :
        comm.Scatterv([vec_global,split_sizes_input, displacements_input,MPI.DOUBLE],vec_local,root=0)
    if dtype == np.int32 :
        #if rank  == 0 : 
            #print (vec_global.shape )
            #print ( split_sizes_input )
            #print ( displacements_input )
        comm.Scatterv([vec_global,split_sizes_input, displacements_input,MPI.INT],vec_local,root=0)
    return vec_local


def gather_4D_array ( vec_local, dtype = np.float64 ):
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
        if dtype == np.float64 :
            comm.Gatherv(vec_local,[vec_global,split_sizes_output,displacements_output,MPI.DOUBLE], root=0) #Gather output data together
        if dtype == np.int32 :
            comm.Gatherv(vec_local,[vec_global,split_sizes_output,displacements_output,MPI.INT], root=0) #Gather output data together
        return vec_global


def allgather_4D_array ( vec_local, dtype = np.float64 ):

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
        if dtype == np.float64 :
            comm.Allgatherv(vec_local,[vec_global,split_sizes_output,displacements_output,MPI.DOUBLE]) #Gather output data together
        if dtype == np.int32 :
            comm.Allgatherv(vec_local,[vec_global,split_sizes_output,displacements_output,MPI.INT]) #Gather output data together
        return vec_global


