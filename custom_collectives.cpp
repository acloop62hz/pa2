#include <iostream> 
#include <mpi.h>

#include "custom_collectives.h"

int custom_many2many(int *send_data, int *sendcounts, int** recv_data_ptr, int rank, int size) {
  /* write your code here */
}

void custom_allreduce_sum(int *local, int *global, int num_elem, int rank, int size) {
  /* write your code here */
}
