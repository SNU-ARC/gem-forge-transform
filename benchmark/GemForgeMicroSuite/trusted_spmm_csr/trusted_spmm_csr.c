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

// yosong, 231016
#define INDEXTYPE uint64_t
#define VALUETYPE float
static const uint64_t num_node			  = 15126;
static const uint64_t num_edge_hvd        = 824617; 
static const uint64_t file_size_indx_hvd  = 2*num_edge_hvd;
static const uint64_t file_size_pntr_hvd  = num_node;
static const uint64_t dim_vector_hvd      = 128;
static const uint64_t nonzero             = 1649234;
static const uint64_t num_iter			  = 1;
//#define CHECK

#define PTTIME
#define PSP

typedef float Value;

// yosong
__attribute__((noinline)) Value trusted_spmm_csr (
   const INDEXTYPE k,      // dimension: col of A and B
   const VALUETYPE *val,   // NNZ value  
   const INDEXTYPE *indx,  // colids -> column indices 
   const INDEXTYPE *pntrb, // starting index for rowptr
   const INDEXTYPE *pntre, // ending index for rowptr
   const VALUETYPE *b,     // Dense B matrix
   const INDEXTYPE ldb,    // leading dimension of B (col size since B row-major)  
   VALUETYPE *c,           // Dense matrix c
   const INDEXTYPE ldc     // leading dimension of c (col size since C row-major) 
) {
   // spmm    
   INDEXTYPE offset_begin = *pntrb;
   INDEXTYPE offset_end = *pntre;
#ifdef PSP
   if (offset_begin < offset_end) {
     __asm__ volatile (
         "stream.input.offset.begin  $0, %[offset_begin] \t\n" // Input stream (offset_begin)
         "stream.input.offset.end  $0, %[offset_end] \t\n"  // Input stream (offset_end)
         "stream.input.ready  $0 \t\n"  // Input stream ready
         :
         :[offset_begin]"r"(offset_begin), [offset_end]"r"(offset_end)
     );
   }
#endif
   for (INDEXTYPE j=offset_begin; j < offset_end; j++) {
//     printf("&b: %x, indx: %lu, offset_begin: %lu, offset_end: %lu.\n", &b[indx[j] * ldb], indx[j], j, offset_end);
     for (INDEXTYPE kk=0; kk < k; kk++)
       c[kk] += (val[j]*b[indx[j]*ldb+kk]);
   }

   return 0;
}

int main(int argc, char **argv) {
  int numThreads = 1;
  printf("argc: %d, argv[1]: %s, argv[2]: %s\n", argc, argv[1], argv[2]);
  if (argc > 1) {
    numThreads = atoi(argv[1]);
  }
  printf("Number of Threads: %d.\n", numThreads);
  omp_set_dynamic(0);
  omp_set_num_threads(numThreads);
  omp_set_schedule(omp_sched_static, 0);
  srand(0);

  // indx from file
  FILE* fp_mtx = fopen("../dataset/graph/harvard/harvard_rows.dat", "rb");
  INDEXTYPE* indx = (INDEXTYPE*) aligned_alloc(CACHE_LINE_SIZE,  file_size_indx_hvd * sizeof(INDEXTYPE));
  if (fp_mtx != NULL) {
    fseek(fp_mtx, 0L, SEEK_END);
    uint64_t sz = ftell(fp_mtx);
    fseek(fp_mtx, 0L, SEEK_SET);
    if (sz == file_size_indx_hvd * sizeof(INDEXTYPE)) {
      fread((void*)indx, sizeof(INDEXTYPE), file_size_indx_hvd, fp_mtx);
    }
    fclose(fp_mtx);
  }
  else {
    printf("Cannot find harvard_row.dat\n");
    return 0;
  }

  //printf("indx[0] = %lu\n" ,indx[0]);
  //printf("indx[1] = %lu\n" ,indx[1]);
  //printf("indx[2] = %lu\n" ,indx[2]);
  //printf("indx[3] = %lu\n" ,indx[3]);

  // pntrb from file
  FILE* fp_mtx2 = fopen("../dataset/graph/harvard/harvard_pntrb.dat", "rb");

  INDEXTYPE* pntrb = (INDEXTYPE*) aligned_alloc(CACHE_LINE_SIZE,  file_size_pntr_hvd * sizeof(INDEXTYPE));

  if (fp_mtx2 != NULL) {
    fseek(fp_mtx2, 0L, SEEK_END);
    uint64_t sz = ftell(fp_mtx2);
    fseek(fp_mtx2, 0L, SEEK_SET);
    if (sz == file_size_pntr_hvd * sizeof(INDEXTYPE)) {
      fread((void*)pntrb, sizeof(INDEXTYPE), file_size_pntr_hvd, fp_mtx2);
    }
    fclose(fp_mtx2);
  }
  else {
    printf("Cannot find harvard_pntrb.dat\n");
    return 0;
  }

  // pntre from file
  FILE* fp_mtx3 = fopen("../dataset/graph/harvard/harvard_pntre.dat", "rb");

  INDEXTYPE* pntre = (INDEXTYPE*) aligned_alloc(CACHE_LINE_SIZE,  file_size_pntr_hvd * sizeof(INDEXTYPE));

  if (fp_mtx3 != NULL) {
    fseek(fp_mtx3, 0L, SEEK_END);
    uint64_t sz = ftell(fp_mtx3);
    fseek(fp_mtx3, 0L, SEEK_SET);
    if (sz == file_size_pntr_hvd * sizeof(INDEXTYPE)) {
      fread((void*)pntre, sizeof(INDEXTYPE), file_size_pntr_hvd, fp_mtx3);
    }
    fclose(fp_mtx3);
  }
  else {
    printf("Cannot find harvard_pntre.dat\n");
    return 0;
  }

  // b alloc
  VALUETYPE* b = (VALUETYPE*) aligned_alloc(CACHE_LINE_SIZE,  num_node*dim_vector_hvd * sizeof(VALUETYPE));
  INDEXTYPE ldb = dim_vector_hvd;

  // b from file
  FILE* fp_mtx4 = fopen("../dataset/graph/harvard/harvard_b_mat.dat", "rb");

  if (fp_mtx4 != NULL) {
    fseek(fp_mtx4, 0L, SEEK_END);
    uint64_t sz = ftell(fp_mtx4);
    fseek(fp_mtx4, 0L, SEEK_SET);
    printf("sz = %lu\n", sz);
    if (sz == num_node*dim_vector_hvd * sizeof(VALUETYPE)) {
      fread((void*)b, sizeof(VALUETYPE), num_node*dim_vector_hvd, fp_mtx4);
    }
    fclose(fp_mtx4);
  }
  else {
    printf("Cannot find harvard_b_mat.dat\n");
    return 0;
  }

  //printf("b[0] = %f\n" ,b[0]);
  //printf("b[1] = %f\n" ,b[1]);
  //printf("b[2] = %f\n" ,b[2]);
  //printf("b[3] = %f\n" ,b[3]);

  // val alloc
  VALUETYPE* val = (VALUETYPE*) aligned_alloc(CACHE_LINE_SIZE,  nonzero * sizeof(VALUETYPE));
  
  // val from file
  FILE* fp_mtx5 = fopen("../dataset/graph/harvard/harvard_val.dat", "rb");

  if (fp_mtx5 != NULL) {
    fseek(fp_mtx5, 0L, SEEK_END);
    uint64_t sz = ftell(fp_mtx5);
    fseek(fp_mtx5, 0L, SEEK_SET);
    printf("sz = %lu\n", sz);
    if (sz == nonzero * sizeof(VALUETYPE)) {
      fread((void*)val, sizeof(VALUETYPE), nonzero, fp_mtx5);
    }
    fclose(fp_mtx5);
  }
  else {
    printf("Cannot find harvard_val.dat\n");
    return 0;
  }

  //printf("val[0] = %f\n" ,val[0]);
  //printf("val[1] = %f\n" ,val[1]);
  //printf("val[2] = %f\n" ,val[2]);
  //printf("val[3] = %f\n" ,val[3]);

  // c alloc
  VALUETYPE* c = (VALUETYPE*) aligned_alloc(CACHE_LINE_SIZE,  num_node*dim_vector_hvd * sizeof(VALUETYPE));
  INDEXTYPE ldc = dim_vector_hvd;

  // m, k
  INDEXTYPE m = num_node;      // rows of A 
  INDEXTYPE k = dim_vector_hvd;// dimension: col of A and B

#ifdef GEM_FORGE
  gf_detail_sim_start();
#endif

#ifdef WARM_CACHE
//  WARM_UP_ARRAY(A, file_size);
//  WARM_UP_ARRAY(B, file_size);
//  WARM_UP_ARRAY(C, file_size);
//  // Initialize the threads.
//#pragma omp parallel for schedule(static) firstprivate(A)
//  for (int tid = 0; tid < numThreads; ++tid) {
//    volatile Value x = *A;
//  }
#endif

#ifdef GEM_FORGE
  gf_reset_stats();
#endif

#ifdef PSP
#ifdef PTTIME 
   #pragma omp parallel for schedule(static)
#endif
  for (uint64_t i = 0; i < numThreads; i++) {
    INDEXTYPE* idx_base_addr = indx;
    uint64_t idx_granularity = sizeof(INDEXTYPE);
    VALUETYPE* val_base_addr = b;
    uint64_t val_granularity = k * sizeof(VALUETYPE);
    __asm__ volatile (
        "stream.cfg.idx.base  $0, %[idx_base_addr] \t\n"    // Configure stream (base address of index)
        "stream.cfg.idx.gran  $0, %[idx_granularity] \t\n"  // Configure stream (access granularity of index)
        "stream.cfg.val.base  $0, %[val_base_addr] \t\n"    // Configure stream (base address of value)
        "stream.cfg.val.gran  $0, %[val_granularity] \t\n"  // Configure stream (access granularity of value)
        "stream.cfg.ready $0 \t\n"  // Configure steam ready
        :
        :[idx_base_addr]"r"(idx_base_addr), [idx_granularity]"r"(idx_granularity),
        [val_base_addr]"r"(val_base_addr), [val_granularity]"r"(val_granularity)
    );
  }
#endif

#ifdef PTTIME 
   #pragma omp parallel for schedule(static)
#endif
  for (uint64_t i = 0; i < m; i++) {
	  trusted_spmm_csr(k, val, indx, &pntrb[i], &pntre[i], b, ldb, &c[i*ldc], ldc);
  }

#ifdef PSP
#ifdef PTTIME 
   #pragma omp parallel for schedule(static)
#endif
  for (uint64_t i = 0; i < numThreads; i++) {
    __asm__ volatile (
        "stream.terminate $0 \t\n"
    );
  }
#endif

#ifdef GEM_FORGE
  gf_detail_sim_end();
#endif

#ifdef CHECK
  //uint64_t err_cnt = 0;
  //Value* C1 = (Value*) aligned_alloc(CACHE_LINE_SIZE, file_size * sizeof(Value));
  //for (uint64_t i = 0; i < num_vector; i++) {
  //  visited[i] = false;
  //}
  //for (uint64_t i = 0; i < num_iter; i++) {
  //  vector_addition_host(A, B, C1, &index_queue[i * num_leaf], sizeof(uint64_t), sizeof(Value) * dim_vector, numThreads);
  //}
  //printf("[ARC-SJ] Starting CHECK stage.\n");
  //for (uint64_t i = 0; i < file_size; i++) {
  //  err_cnt += (C0[i] != C1[i]);
  //}
  //printf("Error count = %ld\n", err_cnt);
#endif

#ifdef CHECK
  //free(C1);
#endif

  free(val);
  free(indx);
  free(pntrb);
  free(pntre);
  free(b);
  free(c);

  return 0;
}
