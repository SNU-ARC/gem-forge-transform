#ifdef GEM_FORGE
#include "gem5/m5ops.h"
#endif

#include <malloc.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "immintrin.h"

typedef float Value;

#define STRIDE 1
#define CHECK
//#define WARM_CACHE

#define N 256
#define M 65536
#define Bx 16
// #define Bx 4
#define By 4

float hsum_ps_sse1(__m128 v) { // v = [ D C | B A ]
  __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1)); // [ C D | A B ]
  __m128 sums = _mm_add_ps(v, shuf); // sums = [ D+C C+D | B+A A+B ]
  shuf = _mm_movehl_ps(shuf, sums);  //  [   C   D | D+C C+D ]  // let the
                                     //  compiler avoid a mov by reusing shuf
  sums = _mm_add_ss(sums, shuf);
  return _mm_cvtss_f32(sums);
}

__attribute__((noinline)) Value foo(Value *A, Value *B, Value *C) {
// #pragma clang loop vectorize(disable)
#pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < N; i += By) {
    Value localSum[By][Bx] = {0};
    for (uint64_t j = 0; j < M; j += Bx) {
#if Bx == 16
      __m512 valB = _mm512_load_ps(B + j);
#elif Bx == 4
      __m128 valB = _mm_load_ps(B + j);
#endif
      for (uint64_t by = 0; by < By; ++by) {
        uint64_t idxA = (i + by) * M + j;
#if Bx == 16
        __m512 valA = _mm512_load_ps(A + idxA);
        __m512 valC = _mm512_load_ps(localSum[by]);
        __m512 valM = _mm512_mul_ps(valA, valB);
        __m512 valS = _mm512_add_ps(valM, valC);
        _mm512_store_ps(localSum[by], valS);
#elif Bx == 4
        __m128 valA = _mm_load_ps(A + idxA);
        __m128 valC = _mm_load_ps(localSum[by]);
        __m128 valM = _mm_mul_ps(valA, valB);
        __m128 valS = _mm_add_ps(valM, valC);
        _mm_store_ps(localSum[by], valS);
#endif
      }
    }
    for (uint64_t by = 0; by < By; ++by) {
#if Bx == 16
      __m512 valS = _mm512_load_ps(localSum[by]);
      Value sum = _mm512_reduce_add_ps(valS);
#elif Bx == 4
      __m128 valS = _mm_load_ps(localSum[by]);
      Value sum = hsum_ps_sse1(valS);
#endif
      C[i + by] = sum;
    }
  }
  return 0.0f;
}

#define CACHE_BLOCK_SIZE 64

int main(int argc, char *argv[]) {

  int numThreads = 1;
  if (argc == 2) {
    numThreads = atoi(argv[1]);
  }
  printf("Number of Threads: %d.\n", numThreads);

  omp_set_dynamic(0);
  omp_set_num_threads(numThreads);
  omp_set_schedule(omp_sched_static, 0);

  Value *A = (Value *)aligned_alloc(CACHE_BLOCK_SIZE, N * M * sizeof(Value));
  Value *B = (Value *)aligned_alloc(CACHE_BLOCK_SIZE, M * sizeof(Value));
  Value *C = (Value *)aligned_alloc(CACHE_BLOCK_SIZE, N * sizeof(Value));

#ifdef GEM_FORGE
  m5_detail_sim_start();
#endif

#ifdef WARM_CACHE
  // This should warm up the cache.
  for (long long i = 0; i < N * M; i += CACHE_BLOCK_SIZE / sizeof(Value)) {
    volatile Value x = A[i];
  }
  for (long long i = 0; i < M; i += CACHE_BLOCK_SIZE / sizeof(Value)) {
    volatile Value x = B[i];
  }
  for (long long i = 0; i < N; i += CACHE_BLOCK_SIZE / sizeof(Value)) {
    volatile Value y = C[i];
  }
  // Start the threads.
#pragma omp parallel for schedule(static)
  for (int tid = 0; tid < numThreads; ++tid) {
    volatile Value x = A[tid];
  }
#endif
#ifdef GEM_FORGE
  m5_reset_stats(0, 0);
#endif

  volatile Value c = foo(A, B, C);
#ifdef GEM_FORGE
  m5_detail_sim_end();
  exit(0);
#endif

#ifdef CHECK
  // Value expected = 0;
  // for (int i = 0; i < N; i += STRIDE) {
  //   expected += a[i];
  // }
  // expected *= NUM_THREADS;
  // printf("Ret = %d, Expected = %d.\n", c, expected);
#endif

  return 0;
}
