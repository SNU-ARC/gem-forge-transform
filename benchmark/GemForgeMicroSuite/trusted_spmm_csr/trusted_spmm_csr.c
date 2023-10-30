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

//#define PSP

typedef float Value;

// Turn off Cache warm-up by default, because it is not our assumption
//#define WARM_CACHE

//__attribute__((noinline)) Value vector_addition_host(Value* A, Value* B, Value* C, uint64_t* index_queue, uint64_t index_granularity, uint64_t value_granularity, int numThreads) {
////  #pragma omp parallel for schedule(static, file_size / numThreads) //firstprivate(A, B, C)
//
//#ifdef PSP
//  // Editor: K16DIABLO (Sungjun Jung)
//  // Example assembly for programmable stream prefetching
//  uint64_t offset_begin = 0;
//  uint64_t offset_end = num_leaf;
//  __asm__ volatile (
//      "stream.cfg.idx.base  $0, %[idx_base_addr] \t\n"    // Configure stream (base address of index)
//      "stream.cfg.idx.gran  $0, %[idx_granularity] \t\n"  // Configure stream (access granularity of index)
//      "stream.cfg.val.base  $0, %[val_base_addr] \t\n"    // Configure stream (base address of value)
//      "stream.cfg.val.gran  $0, %[val_granularity] \t\n"  // Configure stream (access granularity of value)
//      "stream.input.offset.begin  $0, %[offset_begin] \t\n" // Input stream (offset_begin)
//      "stream.input.offset.end  $0, %[offset_end] \t\n"  // Input stream (offset_end)
//      :
//      :[idx_base_addr]"r"(index_queue), [idx_granularity]"r"(index_granularity),
//      [val_base_addr]"r"(A), [val_granularity]"r"(value_granularity),
//      [offset_begin]"r"(offset_begin), [offset_end]"r"(offset_end)
//  );
//#endif
//
//  for (uint64_t i = 0; i < num_leaf; i++) {
//    uint64_t idx = *(index_queue + i);
//    for (uint64_t j = 0; j < dim_vector; j++) {
//      C[idx + j] = A[idx + j] * B[idx + j];
//    }
//  }
//
//#ifdef PSP
//  __asm__ volatile (
//      "stream.terminate $0 \t\n"
//  );
//#endif
//  return 0;
//}

// yosong
//__attribute__((noinline)) void truested_spmm_csr 
__attribute__((noinline)) Value truested_spmm_csr 
(
//   const char tkern,       // kernel variations
   const INDEXTYPE m,      // rows of A 
//   const INDEXTYPE n,      // rows of B
   const INDEXTYPE k,      // dimension: col of A and B
//   const VALUETYPE alpha,  // not used yet  
//   const INDEXTYPE nnz,    // nonzeros  
//   const INDEXTYPE rows,   // number of rows for sparse matrix 
//   const INDEXTYPE cols,   // number of columns for sparse matrix 
   const VALUETYPE *val,   // NNZ value  
   const INDEXTYPE *indx,  // colids -> column indices 
   const INDEXTYPE *pntrb, // starting index for rowptr
   const INDEXTYPE *pntre, // ending index for rowptr
//   const VALUETYPE *a,     // Dense A (X) matrix
//   const INDEXTYPE lda,    // leading dimension of A (col size since A row-major)  
   const VALUETYPE *b,     // Dense B matrix
   const INDEXTYPE ldb,    // leading dimension of B (col size since B row-major)  
//   const VALUETYPE beta,   // beta value 
   VALUETYPE *c,           // Dense matrix c
   const INDEXTYPE ldc     // leading dimension of c (col size since C row-major) 
)
{
#ifdef PTTIME 
   #pragma omp parallel for
#endif
   // spmm    
   for (INDEXTYPE i = 0; i < m; i++)
   {
	  //printf("i = %ld\n", i);
      for (INDEXTYPE j=pntrb[i]; j < pntre[i]; j++)
      {
		 //if(i==0){
			// printf("c[%ld*%ld] += val[%ld]*b[indx[%ld]*%ld]\n", i,ldc,j,j, ldb);
			//intf("k = %ld\n", k);
			//intf("c[%ld*%ld] += val[%ld]*b[%ld*%ld]\n", i,ldc,j,indx[j], ldb);
		 //}
         for (INDEXTYPE kk=0; kk < k; kk++)
            c[i*ldc+kk] += (val[j]*b[indx[j]*ldb+kk]);
      }
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
  FILE* fp_mtx = fopen("../transform/benchmark/GemForgeMicroSuite/trusted_spmm_csr/dataset/harvard_rows.dat", "rb");
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

  //printf("indx[0] = %lu\n" ,indx[0]);
  //printf("indx[1] = %lu\n" ,indx[1]);
  //printf("indx[2] = %lu\n" ,indx[2]);
  //printf("indx[3] = %lu\n" ,indx[3]);

  // pntrb from file
  FILE* fp_mtx2 = fopen("../transform/benchmark/GemForgeMicroSuite/trusted_spmm_csr/dataset/harvard_pntrb.dat", "rb");

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

  //printf("pntrb[0] = %lu\n" ,pntrb[0]);
  //printf("pntrb[1] = %lu\n" ,pntrb[1]);
  //printf("pntrb[2] = %lu\n" ,pntrb[2]);
  //printf("pntrb[3] = %lu\n" ,pntrb[3]);

  // pntre from file
  FILE* fp_mtx3 = fopen("../transform/benchmark/GemForgeMicroSuite/trusted_spmm_csr/dataset/harvard_pntre.dat", "rb");

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

  //printf("pntre[0] = %lu\n" ,pntre[0]);
  //printf("pntre[1] = %lu\n" ,pntre[1]);
  //printf("pntre[2] = %lu\n" ,pntre[2]);
  //printf("pntre[3] = %lu\n" ,pntre[3]);

  // b alloc
  VALUETYPE* b = (VALUETYPE*) aligned_alloc(CACHE_LINE_SIZE,  num_node*dim_vector_hvd * sizeof(VALUETYPE));
  INDEXTYPE ldb = dim_vector_hvd;

  // b from file
  FILE* fp_mtx4 = fopen("../transform/benchmark/GemForgeMicroSuite/trusted_spmm_csr/dataset/harvard_b_mat.dat", "rb");

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

  //printf("b[0] = %f\n" ,b[0]);
  //printf("b[1] = %f\n" ,b[1]);
  //printf("b[2] = %f\n" ,b[2]);
  //printf("b[3] = %f\n" ,b[3]);

  // val alloc
  VALUETYPE* val = (VALUETYPE*) aligned_alloc(CACHE_LINE_SIZE,  nonzero * sizeof(VALUETYPE));
  
  // val from file
  FILE* fp_mtx5 = fopen("../transform/benchmark/GemForgeMicroSuite/trusted_spmm_csr/dataset/harvard_val.dat", "rb");

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

  for (uint64_t i = 0; i < num_iter; i++) {
	  truested_spmm_csr(m, k, val, indx, pntrb, pntre, b, ldb, c, ldc);
  }

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
