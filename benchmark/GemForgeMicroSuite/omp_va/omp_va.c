/**
* @file app.c
* @brief Template for a Host Application Source File.
*
*/
#ifdef GEM_FORGE
#include "gem5/m5ops.h"
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

#define WARM_CACHE
#define CACHE_BLOCK_SIZE 64

static uint32_t *A;
static uint32_t *B;
static uint32_t *C;
static uint32_t *C2; 

/**
* @brief creates a "test file" by filling a buffer of 64MB with pseudo-random values
* @param nr_elements how many 32-bit elements we want the file to be
* @return the buffer address
*/
void  *create_test_file(uint64_t nr_elements) {
    srand(0);
    A = (uint32_t*) aligned_alloc(CACHE_BLOCK_SIZE, nr_elements * sizeof(uint32_t));
    B = (uint32_t*) aligned_alloc(CACHE_BLOCK_SIZE, nr_elements * sizeof(uint32_t));
    C = (uint32_t*) aligned_alloc(CACHE_BLOCK_SIZE, nr_elements * sizeof(uint32_t));
    
//    for (uint64_t i = 0; i < nr_elements; i++) {
//        A[i] = (int) (rand());
//        B[i] = (int) (rand());
//    }

}

/**
* @brief compute output in the host
*/
static void vector_addition_host(uint64_t nr_elements, int t) {
    omp_set_dynamic(0);
    omp_set_num_threads(t);
    omp_set_schedule(omp_sched_static, 0);
    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < nr_elements; i += 16) {
      __m512 valA = _mm512_loadu_epi32(A + i);
      __m512 valB = _mm512_loadu_epi32(B + i);
      __m512 valC = _mm512_add_epi32(valA, valB);
      _mm512_storeu_epi32(C + i, valC);
    }
//    for (uint64_t i = 0; i < nr_elements; i++) {
//        C[i] = A[i] + B[i];
//    }
}

// Params ---------------------------------------------------------------------
typedef struct Params {
    uint64_t   input_size;
    int   n_warmup;
    int   n_reps;
    int   n_threads;
}Params;

void usage() {
    fprintf(stderr,
        "\nUsage:  ./program [options]"
        "\n"
        "\nGeneral options:"
        "\n    -h        help"
        "\n    -t <T>    # of threads (default=8)"
        "\n    -w <W>    # of untimed warmup iterations (default=1)"
        "\n    -e <E>    # of timed repetition iterations (default=3)"
        "\n"
        "\nBenchmark-specific options:"
        "\n    -i <I>    input size (default=8M elements)"
        "\n");
}

struct Params input_params(int argc, char **argv) {
    struct Params p;
    p.input_size    = 16777216;
    p.n_warmup      = 1;
    p.n_reps        = 3;
    p.n_threads     = 1;

    int opt;
    while((opt = getopt(argc, argv, "hi:w:e:t:")) >= 0) {
        switch(opt) {
        case 'h':
        usage();
        exit(0);
        break;
        case 'i': p.input_size    = atoi(optarg); break;
        case 'w': p.n_warmup      = atoi(optarg); break;
        case 'e': p.n_reps        = atoi(optarg); break;
        case 't': p.n_threads        = atoi(optarg); break;
        default:
            fprintf(stderr, "\nUnrecognized option!\n");
            usage();
            exit(0);
        }
    }
    assert(p.n_threads > 0 && "Invalid # of ranks!");

    return p;
}

/**
* @brief Main of the Host Application.
*/
int main(int argc, char **argv) {

//    struct Params p = input_params(argc, argv);
//
//    const uint64_t file_size = p.input_size;

    int numThreads = 1;
    if (argc == 2) {
      numThreads = atoi(argv[1]);
    }
    const uint64_t file_size = 268435456;
//    const uint64_t file_size = 16777216;

#ifdef GEM_FORGE
  m5_detail_sim_start();
#endif

    // Create an input file with arbitrary data.
    create_test_file(file_size);

#ifdef WARM_CACHE
    // This should warm up the cache.
    for (uint64_t i = 0; i < file_size; i += CACHE_BLOCK_SIZE / sizeof(uint32_t)) {
      volatile uint32_t x = A[i];
    }
    for (uint64_t i = 0; i < file_size; i += CACHE_BLOCK_SIZE / sizeof(uint32_t)) {
      volatile uint32_t x = B[i];
    }
    for (uint64_t i = 0; i < file_size; i += CACHE_BLOCK_SIZE / sizeof(uint32_t)) {
      volatile uint32_t y = C[i];
    }
#endif

#ifdef GEM_FORGE
  m5_reset_stats(0, 0);
#endif

//    vector_addition_host(file_size, p.n_threads);
    vector_addition_host(file_size, numThreads);
	
#ifdef GEM_FORGE
  m5_detail_sim_end();
  exit(0);
#endif

    free(A);
    free(B);
    free(C);

   return 0;
 }
