#include <iostream> 
#include <mpi.h>
#include <cmath>
// #include <cstdint>
#include <cstring>
// #include <algorithm>

#include "custom_collectives.h"

void custom_allgather(int *sendcounts_all, int *sendcounts, int rank, int size) {
  memcpy(&sendcounts_all[rank * size], sendcounts, size * sizeof(int));
  for (int j = 0; j < size; j++) {
    if (j == rank)
        continue; // already filled in
    if (rank < j) {
        MPI_Send(sendcounts, size, MPI_INT, j, 0, MPI_COMM_WORLD);
        MPI_Recv(&sendcounts_all[j * size], size, MPI_INT, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else { // rank > j
        MPI_Recv(&sendcounts_all[j * size], size, MPI_INT, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(sendcounts, size, MPI_INT, j, 0, MPI_COMM_WORLD);
    }
  }
}

void custom_alltoall(int *send_buffer, int *receive_buffer, int rank, int global_max, int size) {
  for (int j = 0; j < size; j++) {
    if (j == rank) {
        /* For self communication, just copy locally */
        memcpy(receive_buffer + rank * global_max,
               send_buffer + rank * global_max,
               global_max * sizeof(int));
    } else {
        if (rank < j) {
            MPI_Send(send_buffer + j * global_max, global_max, MPI_INT, j, 1, MPI_COMM_WORLD);
            MPI_Recv(receive_buffer + j * global_max, global_max, MPI_INT, j, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else { // rank > j
            MPI_Recv(receive_buffer + j * global_max, global_max, MPI_INT, j, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(send_buffer + j * global_max, global_max, MPI_INT, j, 1, MPI_COMM_WORLD);
        }
    }
  }
}

int custom_many2many(int *send_data, int *sendcounts, int** recv_data_ptr, int rank, int size) {
  int *sendcounts_all = (int *)malloc(size * size * sizeof(int));
  if (sendcounts_all == NULL) return -1;

  // Use MPI_Allgather instead of custom_allgather
  MPI_Allgather(sendcounts, size, MPI_INT, sendcounts_all, size, MPI_INT, MPI_COMM_WORLD);

  int global_max = 0;
  for (int i = 0; i < size * size; ++i) {
      if (sendcounts_all[i] > global_max) {
          global_max = sendcounts_all[i];
      }
  }

  int *send_displacements = (int *)malloc(size * sizeof(int));
  if (send_displacements == NULL) {
      free(sendcounts_all);
      return -1;
  }
  send_displacements[0] = 0;
  for (int i = 1; i < size; ++i) {
      send_displacements[i] = send_displacements[i-1] + sendcounts[i-1];
  }

  int send_buffer_size = size * global_max;
  int *send_buffer = NULL;
  if (send_buffer_size > 0) {
      send_buffer = (int *)malloc(send_buffer_size * sizeof(int));
      if (send_buffer == NULL) {
          free(sendcounts_all);
          free(send_displacements);
          return -1;
      }
      memset(send_buffer, 0, send_buffer_size * sizeof(int));

      for (int i = 0; i < size; ++i) {
          int src_start = send_displacements[i];
          int dst_start = i * global_max;
          int count = sendcounts[i];
          if (count > 0) {
              memcpy(send_buffer + dst_start, send_data + src_start, count * sizeof(int));
          }
      }
  }

  int *receive_buffer = NULL;
  int receive_buffer_size = size * global_max;
  if (receive_buffer_size > 0) {
      receive_buffer = (int *)malloc(receive_buffer_size * sizeof(int));
      if (receive_buffer == NULL) {
          free(sendcounts_all);
          free(send_displacements);
          free(send_buffer);
          return -1;
      }
  }

  // Use MPI_Alltoallv instead of custom_alltoall
  int *recv_counts = (int *)malloc(size * sizeof(int));
  int *recv_displacements = (int *)malloc(size * sizeof(int));
  if (recv_counts == NULL || recv_displacements == NULL) {
      free(sendcounts_all);
      free(send_displacements);
      free(send_buffer);
      free(receive_buffer);
      free(recv_counts);
      free(recv_displacements);
      return -1;
  }

  for (int i = 0; i < size; ++i) {
      recv_counts[i] = sendcounts_all[i * size + rank];
  }
  recv_displacements[0] = 0;
  for (int i = 1; i < size; ++i) {
      recv_displacements[i] = recv_displacements[i - 1] + recv_counts[i - 1];
  }

  MPI_Alltoallv(send_buffer, sendcounts, send_displacements, MPI_INT,
                receive_buffer, recv_counts, recv_displacements, MPI_INT,
                MPI_COMM_WORLD);

  int total_recv = 0;
  for (int s = 0; s < size; ++s) {
      total_recv += sendcounts_all[s * size + rank];
  }

  *recv_data_ptr = (int *)malloc(total_recv * sizeof(int));
  if (*recv_data_ptr == NULL && total_recv > 0) {
      free(sendcounts_all);
      free(send_displacements);
      free(send_buffer);
      free(receive_buffer);
      free(recv_counts);
      free(recv_displacements);
      return -1;
  }

  if (total_recv > 0 && receive_buffer != NULL) {
      for (int s = 0; s < size; ++s) {
          int count = sendcounts_all[s * size + rank];
          if (count > 0) {
              int src_start = s * global_max;
              memcpy(*recv_data_ptr + recv_displacements[s], receive_buffer + src_start, count * sizeof(int));
          }
      }
  }

  free(sendcounts_all);
  free(send_displacements);
  free(send_buffer);
  free(receive_buffer);
  free(recv_counts);
  free(recv_displacements);

  return total_recv;
}

void custom_allreduce_sum(int *local, int *global, int num_elem, int rank, int size) {
  /* write your code here */
  int dim = ceil(log2(size));
  //Reduce 
  for (int i = 0; i < dim;i++){
    int newRank = (rank ^ (1 << i));
    if (newRank >= size) continue; // dont send out of range
    if ((rank & (1 << i))){
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
