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
//#define LOG

//#define PSP

typedef float Value;

// Turn off Cache warm-up by default, because it is not our assumption
//#define WARM_CACHE

//static const uint64_t file_size = 65536; 
//static const uint64_t file_size = 33554432; 
//static const uint64_t file_size = 16777216;
static uint64_t num_vector;
static uint64_t dim_vector;
static uint64_t num_leaf = 50;
static uint64_t num_iter = 20;
static uint64_t file_size;

__attribute__((noinline)) Value vector_addition_host(Value* A, Value* B, Value* C, uint64_t* index_queue, uint64_t index_granularity, uint64_t value_granularity, int numThreads) {
//  #pragma omp parallel for schedule(static, file_size / numThreads) //firstprivate(A, B, C)

  uint64_t offset_begin = 0;
  uint64_t offset_end = num_leaf;
#ifdef PSP
  // Editor: K16DIABLO (Sungjun Jung)
  // Example assembly for programmable stream prefetching
  __asm__ volatile (
      "stream.cfg.idx.base  $0, %[idx_base_addr] \t\n"    // Configure stream (base address of index)
      "stream.cfg.idx.gran  $0, %[idx_granularity] \t\n"  // Configure stream (access granularity of index)
      "stream.cfg.val.base  $0, %[val_base_addr] \t\n"    // Configure stream (base address of value)
      "stream.cfg.val.gran  $0, %[val_granularity] \t\n"  // Configure stream (access granularity of value)
      "stream.cfg.ready $0 \t\n"  // Configure steam ready
      "stream.input.offset.begin  $0, %[offset_begin] \t\n" // Input stream (offset_begin)
      "stream.input.offset.end  $0, %[offset_end] \t\n"  // Input stream (offset_end)
      "stream.input.ready  $0 \t\n"  // Input stream ready
      :
      :[idx_base_addr]"r"(index_queue), [idx_granularity]"r"(index_granularity),
      [val_base_addr]"r"(A), [val_granularity]"r"(value_granularity),
      [offset_begin]"r"(offset_begin), [offset_end]"r"(offset_end)
  );
#endif

#ifdef LOG
  printf("Indices: ");
  for (uint64_t i = offset_begin; i < offset_end; i++)
    printf("%lu, ", *(index_queue + i));
  printf("\n");
#endif

  for (uint64_t i = offset_begin; i < offset_end; i++) {
    uint64_t idx = *(index_queue + i);
    for (uint64_t j = 0; j < dim_vector; j++) {
      C[idx * dim_vector + j] = A[idx * dim_vector + j] * B[idx * dim_vector + j];
    }
  }

#ifdef PSP
  __asm__ volatile (
      "stream.terminate $0 \t\n"
  );
#endif
  return 0;
}

int main(int argc, char **argv) {
  int numThreads = 1;
  if (argc > 1) {
    numThreads = atoi(argv[1]);
  }
  printf("Number of Threads: %d.\n", numThreads);
  omp_set_dynamic(0);
  omp_set_num_threads(numThreads);
  omp_set_schedule(omp_sched_static, 0);
  srand(0);

  // Open file and parse feature vector num & dim
  FILE* input_file = fopen(argv[2], "rb");
  if (input_file == NULL) {
    printf("[Error] Invalid file name\n");
    return 0;
  }
  fread((void*)&num_vector, sizeof(uint64_t), 1, input_file);
  fread((void*)&dim_vector, sizeof(uint64_t), 1, input_file);
  file_size = num_vector * dim_vector;
  fseek(input_file, 0L, SEEK_SET);
  printf("num_vector: %lu, dim_vector: %lu, file_size: %lu\n", num_vector, dim_vector, file_size);

  // Create an input file with arbitrary data.
  Value* data = (Value*) aligned_alloc(CACHE_LINE_SIZE, 2 * file_size * sizeof(Value));
  Value* A = data;
  Value* B = data + file_size;
  Value* C0 = (Value*) aligned_alloc(CACHE_LINE_SIZE, file_size * sizeof(Value));

  // Read data from file
  fseek(input_file, 2 * sizeof(uint64_t), SEEK_CUR);
  fread((void*)A, sizeof(Value), file_size, input_file);
  fseek(input_file, 0L, SEEK_SET);
  fseek(input_file, 2 * sizeof(uint64_t), SEEK_CUR);
  fread((void*)B, sizeof(Value), file_size, input_file);
  fclose(input_file);

  uint64_t* index_queue = (uint64_t*) aligned_alloc(CACHE_LINE_SIZE, num_iter * num_leaf * sizeof(uint64_t));
  for (uint64_t i = 0; i < num_iter * num_leaf; i++)
    index_queue[i] = rand() % num_vector;

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

#pragma omp parallel for schedule(static, num_iter / numThreads) //firstprivate(A, B, C)
  for (uint64_t i = 0; i < num_iter; i++) {
    vector_addition_host(A, B, C0, &index_queue[rand() % num_iter * num_leaf], sizeof(uint64_t), sizeof(Value) * dim_vector, numThreads);
  }

#ifdef GEM_FORGE
  gf_detail_sim_end();
#endif

#ifdef CHECK
  uint64_t err_cnt = 0;
  Value* C1 = (Value*) aligned_alloc(CACHE_LINE_SIZE, file_size * sizeof(Value));
  for (uint64_t i = 0; i < num_vector; i++) {
    visited[i] = false;
  }
  for (uint64_t i = 0; i < num_iter; i++) {
    vector_addition_host(A, B, C1, &index_queue[i * num_leaf], sizeof(uint64_t), sizeof(Value) * dim_vector, numThreads);
  }
  printf("[ARC-SJ] Starting CHECK stage.\n");
  for (uint64_t i = 0; i < file_size; i++) {
    err_cnt += (C0[i] != C1[i]);
  }
  printf("Error count = %ld\n", err_cnt);
#endif

  free(data);
  free(C0);
#ifdef CHECK
  free(C1);
#endif
  free(index_queue);

  return 0;
}
