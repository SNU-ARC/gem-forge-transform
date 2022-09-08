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

typedef uint32_t Value;

// Turn off Cache warm-up by default, because it is not our assumption
//#define WARM_CACHE

//static const uint64_t file_size = 65536; 
static const uint64_t file_size = 33554432; 
//static const uint64_t file_size = 16777216;

__attribute__((noinline)) Value vector_addition_host(Value* A, Value* B, Value* C, int numThreads) {
  #pragma omp parallel for schedule(static, file_size / numThreads) firstprivate(A, B, C)
    for (uint64_t i = 0; i < file_size; i += 16) {
      __m512i valA = _mm512_loadu_epi32(A + i);
      __m512i valB = _mm512_loadu_epi32(B + i);
      __m512i valC = _mm512_add_epi32(valA, valB);
      _mm512_storeu_epi32(C + i, valC);
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

  // Create an input file with arbitrary data.
  Value* A = (Value*) aligned_alloc(CACHE_LINE_SIZE, file_size * sizeof(Value));
  Value* B = (Value*) aligned_alloc(CACHE_LINE_SIZE, file_size * sizeof(Value));
  Value* C = (Value*) aligned_alloc(CACHE_LINE_SIZE, file_size * sizeof(Value));

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

  vector_addition_host(A, B, C, numThreads);

#ifdef GEM_FORGE
  gf_detail_sim_end();
  exit(0);
#endif

  free(A);
  free(B);
  free(C);

  return 0;
}
