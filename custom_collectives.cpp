#include <iostream> 
#include <mpi.h>
#include <cmath>
// #include <cstdint>
#include <cstring>
// #include <algorithm>

#include "custom_collectives.h"

int custom_many2many(int *send_data, int *sendcounts, int** recv_data_ptr, int rank, int size) {
  /* write your code here */
  return 0;
}

void custom_allreduce_sum(int *local, int *global, int num_elem, int rank, int size) {
  /* write your code here */
  int dim = ceil(log2(size));
  //Reduce 
  for (int i = 0; i < dim;i++){
    int newRank = (rank ^ 1 << i);
    if (newRank >= size) continue; // dont send out of range
    if ((rank & 1 << i)){
      MPI_Send(local, num_elem, MPI_INT, newRank, 0, MPI_COMM_WORLD);
    }else{
      int recv[num_elem]; 
      MPI_Recv(recv,num_elem,MPI_INT,newRank,MPI_ANY_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      for (int j=0; j<num_elem; j++){
        local[j]+=recv[j];
      }
    }
  }

  //broadcast
  int flip = 1 << (dim-1);
  int mask = flip-1;
  for (int i = 0; i<dim; i++){
    int newRank = (rank ^ flip);
    if ((rank & mask) == 0 && newRank < size ){
      if (rank & flip){
        MPI_Recv(local,num_elem,MPI_INT, newRank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      }else {
        MPI_Send(local,num_elem,MPI_INT, newRank,0,MPI_COMM_WORLD);
      }
    }
    mask >>= 1;
    flip >>= 1;
  }

  // printf("rank: %d\n",rank);
  // for (int j=0; j<num_elem; j++){
  //   printf("%d ",local[j]);
  // }
  // printf("\n");

  if (rank == 0){
    // std::copy(local,local+num_elem,global);
    std::memcpy(global,local,sizeof(int)*num_elem);
    // printf("global: \n");
    // for (int j=0; j<num_elem; j++){
    //   printf("%d ",global[j]);
    // }
  }

}
