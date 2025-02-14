/**
 * Simple dense vector add.
 */

#include "gfm_utils.h"
#include <math.h>
#include <stdio.h>

#include "immintrin.h"

typedef float Value;

#define STRIDE 1
#define CHECK
#define WARM_CACHE

__attribute__((noinline)) Value foo(Value *a, int N) {
  __m512 valS = _mm512_set1_ps(2.0f);
#pragma clang loop vectorize(disable) unroll(disable)
  for (int i = 0; i < N; i += 16) {
    _mm512_store_ps(a + i, valS);
  }
  return 0;
}

// 65536*2*4 = 512kB.
const int N = 65536 * 2;
Value a[N];

int main() {

  gf_detail_sim_start();
#ifdef WARM_CACHE
  // This should warm up the cache.
  for (long long i = 0; i < N; i++) {
    a[i] = 1;
  }
#endif
  gf_reset_stats();
  volatile Value ret = foo(a, N);
  gf_detail_sim_end();

#ifdef CHECK
  Value expected = N * 2;
  Value computed = 0;
  printf("Start computed.\n");
#pragma clang loop vectorize(disable) unroll(disable)
  for (int i = 0; i < N; i += STRIDE) {
    Value v = a[i];
    computed += v;
  }
  printf("Computed = %f, Expected = %f.\n", computed, expected);
  // This result should be extactly the same.
  if (computed != expected) {
    gf_panic();
  }
#endif

  return 0;
}
