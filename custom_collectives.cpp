#include <iostream> 
#include <mpi.h>
#include <math.h>

#include "custom_collectives.h"

int custom_many2many(int *send_data, int *sendcounts, int** recv_data_ptr, int rank, int size) {
  /* write your code here */
}

void custom_allreduce_sum(int *local, int *global, int num_elem, int rank, int size) {
  /* write your code here */

  for (int i = 0; i < num_elem; i++) {
    global[i] = local[i];
  }

  int dim = (int)ceil(log2(size));
  int partner;

  for(int j = 0; j < dim; j++){

    partner = rank ^ (1 << j); // partner id may be out of range if size is not 2^n

    int received_sum[num_elem];


    MPI_Sendrecv(global, num_elem, MPI_INT, partner, 0,
      received_sum, num_elem, MPI_INT, partner, 0,
      MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    

    for(int i = 0; i < num_elem; i++){
      global[i] += received_sum[i];
    }
  }

  // print the results
  if(rank ==0){
    for(int i = 0; i < num_elem; i++){
      printf("%d ", global[i]);
    }
  }
  
    


}
