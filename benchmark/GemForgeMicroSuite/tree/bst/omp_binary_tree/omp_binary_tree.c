#include "gfm_utils.h"

#ifndef NO_OMP
#include <omp.h>
#endif
#include <stdlib.h>

#include "../bst.h"

#define WARM_CACHE

typedef int64_t Value;

void foo_warm(struct BSTreeNode *root, Value *keys, int64_t totalKeys,
              uint8_t *matched) {
  for (int64_t i = 0; i < totalKeys; ++i) {
    Value key = keys[i];
    struct BSTreeNode *node = root;
    uint8_t found = 0;
    Value val = 0;
    do {
      val = node->val;
      struct BSTreeNode *lhs = node->lhs;
      struct BSTreeNode *rhs = node->rhs;
      node = (val > key) ? lhs : rhs;
      found = found || (val == key);
    } while (val != key && node != NULL);
    matched[i] = found;
  }
}

__attribute__((noinline)) void foo(struct BSTreeNode *root, Value *keys,
                                   int64_t totalKeys, uint8_t *matched) {
#ifndef NO_OMP
#pragma omp parallel for schedule(static) firstprivate(root, keys, matched)
#endif
  for (int64_t i = 0; i < totalKeys; ++i) {
    Value key = keys[i];
    struct BSTreeNode *node = root;
    uint8_t found = 0;
    Value val = 0;
    if (node) {
      do {
        val = node->val;
        struct BSTreeNode *lhs = node->lhs;
        struct BSTreeNode *rhs = node->rhs;
        node = (val > key) ? lhs : rhs;
        found = found || (val == key);
      } while (val != key && node != NULL);
    }
    matched[i] = found;
  }
}

int main(int argc, char *argv[]) {

  int numThreads = 1;
  uint64_t totalElements = 512 * 1024;
  uint64_t totalKeys = 8 * 1024 * 1024;
  uint64_t hitRatio = 8;
  int check = 1;
  if (argc >= 2) {
    numThreads = atoi(argv[1]);
  }
  if (argc >= 3) {
    totalElements = atoll(argv[2]);
  }
  if (argc >= 4) {
    totalKeys = atoll(argv[3]);
  }
  if (argc >= 5) {
    hitRatio = atoll(argv[4]);
  }
  if (argc >= 6) {
    check = atoi(argv[5]);
  }

  printf("NumThreads %d. TotalElm %lu. Tree %lu kB/Thread.\n", numThreads,
         totalElements,
         totalElements * sizeof(struct BSTreeNode) / numThreads / 1024);
  printf("TotalKeys %lu. HitRatio %lu. Key %lu kB/Thread.\n", totalKeys,
         hitRatio, totalKeys * sizeof(Value) / numThreads / 1024);

#ifndef NO_OMP
  omp_set_dynamic(0);
  omp_set_num_threads(numThreads);
  omp_set_schedule(omp_sched_static, 0);
#endif

  Value keyMax = totalElements * hitRatio;
  struct BSTree tree = generateUniformTree(keyMax, totalElements);

  // Generate the keys.
  Value *keys = aligned_alloc(64, sizeof(Value) * totalKeys);
  for (int64_t i = 0; i < totalKeys; ++i) {
    keys[i] = (int64_t)(((float)(rand()) / (float)(RAND_MAX)) * keyMax);
  }

  // Allocated the matched results.
  uint8_t *matched = aligned_alloc(64, sizeof(uint8_t) * totalKeys);

  gf_detail_sim_start();

#ifdef WARM_CACHE
  WARM_UP_ARRAY(tree.array, totalElements);
  WARM_UP_ARRAY(keys, totalKeys);
  WARM_UP_ARRAY(matched, totalKeys);
#pragma omp parallel for schedule(static)
  for (int tid = 0; tid < numThreads; ++tid) {
    volatile Value x = keys[tid];
  }

#endif

  gf_reset_stats();
  foo(tree.root, keys, totalKeys, matched);
  gf_detail_sim_end();

  if (check) {
    uint8_t *expected_matched = aligned_alloc(64, sizeof(uint8_t) * totalKeys);
    foo_warm(tree.root, keys, totalKeys, expected_matched);
    uint64_t totalHits = 0;
    for (int64_t i = 0; i < totalKeys; ++i) {
      Value key = keys[i];
      uint8_t expected = expected_matched[i];
      uint8_t found = matched[i];
      if (found) {
        totalHits++;
      }
      if (found != expected) {
        printf("Mismatch %ldth Key = %ld. Expected %d != Ret %d. TotalElements "
               "%lu.\n",
               i, key, expected, found, totalElements);
        gf_panic();
      }
    }
    printf("All matched. TotalHits %lu.\n", totalHits);
  } else {
    printf("No check.\n");
  }
  return 0;
}
