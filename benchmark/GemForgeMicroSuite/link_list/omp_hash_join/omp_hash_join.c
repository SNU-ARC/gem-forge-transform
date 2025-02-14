// Search in link list.
#include "gfm_utils.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

typedef int64_t Value;

struct Node {
  struct Node *next;
  Value val;
};

__attribute__((noinline)) void foo(struct Node **heads, uint64_t hashMask,
                                   Value *keys, int64_t totalKeys,
                                   uint8_t *matched) {

#pragma omp parallel for schedule(static)                                      \
    firstprivate(heads, hashMask, keys, matched)
  for (int64_t i = 0; i < totalKeys; ++i) {
#pragma ss stream_name "gfm.omp_hash_join.key.ld"
    Value key = keys[i];
    Value hash = key & hashMask;
#pragma ss stream_name "gfm.omp_hash_join.head.ld"
    struct Node *head = heads[hash];
    Value val = 0;
    uint8_t found = 0;
    if (head) {
      /**
       * This helps us getting a single BB loop body, with simple condition.
       */
      do {
#pragma ss stream_name "gfm.omp_hash_join.val.ld"
        Value v = head->val;

#pragma ss stream_name "gfm.omp_hash_join.next.ld"
        struct Node *n = head->next;

        val = v;
        head = n;
        // This ensures that found is a reduction variable.
        found = found || (val == key);
      } while (val != key && head != NULL);
    }
#pragma ss stream_name "gfm.omp_hash_join.match.st"
    matched[i] = found;
  }
}

struct InputArgs {
  int numThreads;
  uint64_t totalElements;
  uint64_t elementsPerBucket;
  uint64_t totalKeys;
  uint64_t hitRatio;
  int check;
  int warm;
};

struct InputArgs parseArgs(int argc, char *argv[]) {

  struct InputArgs args;

  args.numThreads = 1;
  args.totalElements = 512 * 1024;
  args.elementsPerBucket = 8;
  args.totalKeys = 8 * 1024 * 1024;
  args.hitRatio = 8;
  args.check = 0;
  args.warm = 0;

  int argx = 2;
  if (argc >= argx) {
    args.numThreads = atoi(argv[argx - 1]);
  }
  argx++;
  if (argc >= argx) {
    args.totalElements = atoll(argv[argx - 1]);
  }
  argx++;
  if (argc >= argx) {
    args.elementsPerBucket = atoll(argv[argx - 1]);
  }
  argx++;
  if (argc >= argx) {
    args.totalKeys = atoll(argv[argx - 1]);
  }
  argx++;
  if (argc >= argx) {
    args.hitRatio = atoll(argv[argx - 1]);
  }
  argx++;
  if (argc >= argx) {
    args.check = atoi(argv[argx - 1]);
  }
  argx++;
  if (argc >= argx) {
    args.warm = atoi(argv[argx - 1]);
  }
  argx++;

  return args;
}

const int MAX_FILE_NAME = 256;
char fileName[MAX_FILE_NAME];

const char *formatBytes(uint64_t *bytes) {
  const char *suffix = "B";
  if ((*bytes) >= 1024 * 1024) {
    (*bytes) /= 1024 * 1024;
    suffix = "MB";
  } else if ((*bytes) >= 1024) {
    (*bytes) /= 1024;
    suffix = "kB";
  }
  return suffix;
}

const char *generateFileName(struct InputArgs args) {

  uint64_t totalElementBytes = args.totalElements * sizeof(struct Node);
  const char *elementBytesSuffix = formatBytes(&totalElementBytes);

  uint64_t totalKeyBytes = args.totalKeys * sizeof(Value);
  const char *keySuffix = formatBytes(&totalKeyBytes);

  snprintf(fileName, MAX_FILE_NAME, "hash_join_%lu%s_%lu_%lu%s_%lu.data",
           totalElementBytes, elementBytesSuffix, args.elementsPerBucket,
           totalKeyBytes, keySuffix, args.hitRatio);

  return fileName;
}

struct DataArrays {
  struct Node *nodes;
  struct Node **heads;
  Value *keys;
};

void dumpData(struct DataArrays data, struct InputArgs args,
              const char *fileName) {
  const uint64_t totalBuckets = args.totalElements / args.elementsPerBucket;
  FILE *f = fopen(fileName, "wb");
  if (!f) {
    gf_panic();
  }
  /**
   * Format:
   * 1. Address of nodes.
   * 2. binary dumping each array.
   */
  fwrite(&data.nodes, sizeof(data.nodes), 1, f);
  fwrite(data.nodes, sizeof(struct Node), args.totalElements, f);
  fwrite(data.heads, sizeof(struct Node *), totalBuckets, f);
  fwrite(data.keys, sizeof(Value), args.totalKeys, f);
  fclose(f);
}

struct DataArrays loadData(struct InputArgs args, const char *fileName) {
  struct DataArrays data;
  data.nodes = NULL;
  data.heads = NULL;
  data.keys = NULL;

  FILE *f = fopen(fileName, "rb");
  if (f) {

    const uint64_t totalBuckets = args.totalElements / args.elementsPerBucket;

    struct Node *oldNodesPtr = NULL;
    fread(&oldNodesPtr, sizeof(oldNodesPtr), 1, f);

    data.nodes =
        aligned_alloc(PAGE_SIZE, sizeof(struct Node) * args.totalElements);
    data.heads = aligned_alloc(PAGE_SIZE, sizeof(struct Node *) * totalBuckets);
    data.keys = aligned_alloc(PAGE_SIZE, sizeof(Value) * args.totalKeys);

    fread(data.nodes, sizeof(struct Node), args.totalElements, f);
    fread(data.heads, sizeof(struct Node *), totalBuckets, f);
    fread(data.keys, sizeof(Value), args.totalKeys, f);

    /**
     * Fix all the pointers.
     */
    for (uint64_t i = 0; i < args.totalElements; ++i) {
      if (data.nodes[i].next != NULL) {
        data.nodes[i].next = data.nodes + (data.nodes[i].next - oldNodesPtr);
      }
    }
    for (uint64_t i = 0; i < totalBuckets; ++i) {
      if (data.heads[i] != NULL) {
        data.heads[i] = data.nodes + (data.heads[i] - oldNodesPtr);
      }
    }
    printf("Loaded data from file %s.\n", fileName);

    fclose(f);
  }

  return data;
}

struct DataArrays generateData(struct InputArgs args) {
  const uint64_t totalBuckets = args.totalElements / args.elementsPerBucket;
  const char *fileName = generateFileName(args);

  {
    struct DataArrays data = loadData(args, fileName);
    if (data.nodes != NULL) {
      return data;
    }
  }

  /**
   * Interesting fact: creating the linked list via
   * a sequence of malloc of the same size will actually
   * result in a lot of locality. The difference between
   * the consecutive allocated addresses is always 32 in
   * this example.
   */
  printf("Start to generate data.\n");
  uint64_t totalNodes = args.totalElements;
  struct Node *nodes =
      aligned_alloc(PAGE_SIZE, sizeof(struct Node) * totalNodes);
  struct Node **nodes_ptr =
      aligned_alloc(PAGE_SIZE, sizeof(struct Node *) * totalNodes);
  struct Node **heads =
      aligned_alloc(PAGE_SIZE, sizeof(struct Node *) * totalBuckets);

  for (uint64_t i = 0; i < totalNodes; ++i) {
    nodes[i].next = NULL;
    nodes_ptr[i] = &nodes[i];
  }

  // Shuffle the nodes.
  for (int j = totalNodes - 1; j > 0; --j) {
    int i = (int)(((float)(rand()) / (float)(RAND_MAX)) * j);
    struct Node *tmp = nodes_ptr[i];
    nodes_ptr[i] = nodes_ptr[j];
    nodes_ptr[j] = tmp;
  }

  // Connect the nodes.
  for (int64_t j = 0; j < totalBuckets; ++j) {
    heads[j] = nodes_ptr[j * args.elementsPerBucket];
    for (int64_t i = 0; i < args.elementsPerBucket; ++i) {
      int64_t idx = j * args.elementsPerBucket + i;
      if (i < args.elementsPerBucket - 1) {
        // Connect.
        nodes_ptr[idx]->next = nodes_ptr[idx + 1];
      }
      // Set one value with the correct hash values.
      Value val = i * totalBuckets + j;
      nodes_ptr[idx]->val = val;
    }
  }

  // Generate the keys.
  Value *keys = aligned_alloc(64, sizeof(Value) * args.totalKeys);
  Value keyMax = args.totalElements * args.hitRatio;
  for (int64_t i = 0; i < args.totalKeys; ++i) {
    keys[i] = (int64_t)(((float)(rand()) / (float)(RAND_MAX)) * keyMax);
  }

  struct DataArrays data;
  data.nodes = nodes;
  data.heads = heads;
  data.keys = keys;

  dumpData(data, args, fileName);

  return data;
}

int main(int argc, char *argv[]) {

  struct InputArgs args = parseArgs(argc, argv);

  const uint64_t totalBuckets = args.totalElements / args.elementsPerBucket;
  if (totalBuckets & (totalBuckets - 1)) {
    printf("TotalBucket %lu must be power of 2.\n", totalBuckets);
    gf_panic();
  }
  const uint64_t hashMask = totalBuckets - 1;
  printf("NumThreads %d. TotalElm %lu. ElmPerBkt %lu. NumBkt %lu. Elm %lu "
         "kB/Thread.\n",
         args.numThreads, args.totalElements, args.elementsPerBucket,
         totalBuckets,
         args.totalElements * sizeof(struct Node) / args.numThreads / 1024);
  printf("TotalKeys %lu. HitRatio %lu. Key %lu kB/Thread.\n", args.totalKeys,
         args.hitRatio,
         args.totalKeys * sizeof(Value) / args.numThreads / 1024);

  omp_set_dynamic(0);
  omp_set_num_threads(args.numThreads);
  omp_set_schedule(omp_sched_static, 0);

  struct DataArrays data = generateData(args);

  // Allocated the matched results.
  uint8_t *matched = aligned_alloc(64, sizeof(uint8_t) * args.totalKeys);

  gf_detail_sim_start();
  if (args.warm) {
    WARM_UP_ARRAY(data.nodes, args.totalElements);
    WARM_UP_ARRAY(data.keys, args.totalKeys);
    WARM_UP_ARRAY(matched, args.totalKeys);
  }

  Value p;
  Value *pp = &p;
#pragma omp parallel for schedule(static)
  for (int tid = 0; tid < args.numThreads; ++tid) {
    volatile Value x = *pp;
  }

  gf_reset_stats();
  foo(data.heads, hashMask, data.keys, args.totalKeys, matched);
  gf_detail_sim_end();

  if (args.check) {
    for (int64_t i = 0; i < args.totalKeys; ++i) {
      Value key = data.keys[i];
      uint8_t expected = (key < args.totalElements);
      uint8_t found = matched[i];
      if (found != expected) {
        printf("Mismatch %ldth Key = %ld. Expected %d != Ret %d. TotalElements "
               "%lu.\n",
               i, key, expected, found, args.totalElements);
        gf_panic();
      }
    }
    printf("All matched.\n");
  } else {
    printf("No check.\n");
  }
  return 0;
}