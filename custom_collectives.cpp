#include "custom_collectives.h"

#include <mpi.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>


void Custom_Scatter(int* sendbuf, int sendcount, MPI_Datatype sendtype,
                    int* recvbuf, int recvcount, MPI_Datatype recvtype,
                    int root, MPI_Comm comm) {
   //Implement the code below

    return;
}

void Custom_Allgather(int* sendbuf, int sendcount, MPI_Datatype sendtype,
                      int* recvbuf, int recvcount, MPI_Datatype recvtype,
                      MPI_Comm comm) {
    //Implement the code below

    return;
}


void Custom_Allreduce(int* sendbuf, int* recvbuf, int count,
                      MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
    //Implement the code below
    //op will be MPI_SUM

    return;
}

void Custom_Alltoall_Hypercube(int* sendbuf, int sendcount, MPI_Datatype sendtype,
                               int* recvbuf, int recvcount, MPI_Datatype recvtype,
                               MPI_Comm comm) {
   //Implement the code below


   return;
}


void Custom_Alltoall_Arbitrary(int* sendbuf, int sendcount, MPI_Datatype sendtype,
                               int* recvbuf, int recvcount, MPI_Datatype recvtype,
                               MPI_Comm comm) {
    //Implement the code below

    return;
}