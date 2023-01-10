#ifdef GEM_FORGE
#include "../gfm_utils.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <stdint.h>

#include <omp.h>
#include "immintrin.h"

//#define CHECK

typedef float Value;

// Turn off Cache warm-up by default, because it is not our assumption
//#define WARM_CACHE

//static const uint64_t file_size = 65536; 
//static const uint64_t file_size = 33554432; 
//static const uint64_t file_size = 16777216;
static const uint64_t num_vector = 1000000;
static const uint64_t dim_vector = 96;
static const uint64_t num_leaf = 50;
static const uint64_t num_iter = 200;
static const uint64_t file_size = num_vector * dim_vector;

__attribute__((noinline)) Value vector_addition_host(void* data, uint64_t* index_queue, bool* visited, int numThreads) {
  #pragma omp parallel for schedule(static, file_size / numThreads) //firstprivate(A, B, C)
  for (uint64_t i = 0; i < num_leaf; i++) {
    uint64_t idx = *(index_queue + i);
//    if (idx % 4 < 0) continue;
    if (visited[idx]) continue;
    Value* A = (Value*)(data + idx * sizeof(Value));
    Value* B = (Value*)(data + (file_size + idx) * sizeof(Value));
    Value* C = (Value*)(data + (2 * file_size + idx) * sizeof(Value));
    for (uint64_t j = 0; j < dim_vector; j++) {
      C[idx + j] = A[idx + j] * B[idx + j];
    }
//    for (uint64_t j = 1; j < dim_vector; j++) {
//      C[idx] = C[idx + j];
//    }
    visited[idx] = true;
  }
  return 0;
}

int main(int argc, char **argv) {

  int numThreads = 1;
  if (argc == 2) {
    numThreads = atoi(argv[1]);
  }
  printf("Number of Threads: %d.\n", numThreads);
  omp_set_dynamic(0);
  omp_set_num_threads(numThreads);
  omp_set_schedule(omp_sched_static, 0);
  srand(0);

  // Create an input file with arbitrary data.
  void* data = (void*) aligned_alloc(CACHE_LINE_SIZE, 3 * file_size * sizeof(Value));
  uint64_t* index_queue = (uint64_t*) aligned_alloc(CACHE_LINE_SIZE, num_iter * num_leaf * sizeof(uint64_t));
  for (uint64_t i = 0; i < num_iter * num_leaf; i++)
    index_queue[i] = rand() / num_vector;
  bool* visited = (bool*) aligned_alloc(CACHE_LINE_SIZE, num_vector * sizeof(bool));
  for (uint64_t i = 0; i < num_vector; i++) {
    visited[i] = false; // (i % 4 < 0) ? false : true;
//    if (i % num_vector == 0)
//      printf("%lu\n", i);
  }

#ifdef GEM_FORGE
  gf_detail_sim_start();
#endif

#ifdef WARM_CACHE
  WARM_UP_ARRAY(A, file_size);
  WARM_UP_ARRAY(B, file_size);
  WARM_UP_ARRAY(C, file_size);
  // Initialize the threads.
#pragma omp parallel for schedule(static) firstprivate(A)
  for (int tid = 0; tid < numThreads; ++tid) {
    volatile Value x = *A;
  }
#endif

#ifdef GEM_FORGE
  gf_reset_stats();
#endif

  for (uint64_t i = 0; i < num_iter; i++) {
    vector_addition_host(data, &index_queue[rand() % num_iter * num_leaf], visited, numThreads);
  }

#ifdef CHECK
  uint64_t err_cnt = 0;
  Value* C_CHECK = (Value*) aligned_alloc(CACHE_LINE_SIZE, file_size * sizeof(Value));
  vector_addition_host(A, B, C_CHECK, index_queue, numThreads);
  for (uint64_t i = 0; i < file_size; i++) {
    err_cnt += (C[i] == C_CHECK[i]);
  }
  printf("Error count = %ld\n", err_cnt);
#endif

#ifdef GEM_FORGE
  gf_detail_sim_end();
  exit(0);
#endif

  free(data);
//  free(A);
//  free(B);
//  free(C);
  free(index_queue);
//  free(visited);

  return 0;
}
