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

#include<stdint.h>
#ifdef PTTIME
   #include<omp.h>
#endif
#define SREAL 1
#include "simd.h"
//"/home2/youngook/research/01_NDP/01_ISCA_23/FusedMM/kernels/simd/simd.h"
#if VLEN != 8
   #error "ARCH VLEN doesn't match with generator's VLEN, see simd.h " 
#endif

#define BETA0
/*
 * Register block  A,C and B(innermost loop) will require most registers, works 
 * better on small value of k
 */
#ifdef BETA0 
//void sgfusedMM_K128_spmm_b0_csr
__attribute__((noinline)) Value sgfusedMM_K128_spmm_b0_csr
#else /* BETA1 version */
//void sgfusedMM_K128_spmm_b1_csr
__attribute__((noinline)) Value sgfusedMM_K128_spmm_b1_csr
#endif
(
//   const char tkern,  	   // 's' 't' 'm'
   const INDEXTYPE m,      // rows of dense A matrix 
//   const INDEXTYPE n,      // rows of dense B matrix
   const INDEXTYPE k,      // cols of A or dimension. not used since K compile time   
//   const float alpha,     // const to scale, not use yet  
//   const INDEXTYPE nnz,    // nonzeros of the sparse matrix 
//   const INDEXTYPE rows,   // number of rows of the sparse matrix  
//   const INDEXTYPE cols,   // number of columns of the sparse matrix 
   const float *val,       // value of  the sparse matrix 
   const INDEXTYPE *indx,  // colids -> column indices of sparse matrix 
   const INDEXTYPE *pntrb, // starting index for rowptr of csr of sparse matrix
   const INDEXTYPE *pntre, // ending index for rowptr of csr of sparse matrix 
// const float *a,        // Dense A matrix
// const INDEXTYPE lda,    // leading dimension of a (col size since row-major)  
   const float *b,        // Dense B matrix
   const INDEXTYPE ldb,    // leading dimension of b (col size since row-major)  
// const float beta,      // beta value, compile time not used  
   float *c,              // Dense matrix c
   const INDEXTYPE ldc     // leading dimension size of c (col size since roa-major) 
)
{
#if defined(PTTIME) && defined(LDB)
   omp_set_num_threads(NTHREADS);
   #pragma omp parallel
   {
      INDEXTYPE RowPerThd, tt;
      INDEXTYPE i, rowb, rowe;
      INDEXTYPE Mnnz = 0; /* non-zero count in M rows  */
      INDEXTYPE deg, cumRow, curRow;
      INDEXTYPE id = omp_get_thread_num();
      INDEXTYPE nthreads = omp_get_num_threads(); 
      
      for (i=0; i < m; i++)
         Mnnz += (pntre[i] - pntrb[i]); 
      RowPerThd = Mnnz / nthreads; 
      
      curRow = cumRow = 0; 
      tt = 1; 
      rowe = -1;  /* init */
      /* set rowstart for 1st thread */ 
      if (id == 0) 
         rowb = 0;
      for (i=0; i < m; i++)
      {
         deg = pntre[i] - pntrb[i]; 
         cumRow += deg;
         curRow += deg;
         if (curRow > RowPerThd)
         {
            if (tt == id)
               rowb = i; 
            else if (tt == id+1)
               rowe = i; 
            curRow = 0;
            RowPerThd = (Mnnz - cumRow) / (nthreads - tt);
            tt += 1; 
         }
      }
      if (tt == id+1)
         rowe = m; 

      for (i=rowb; i < rowe; i++)
#else /* not LBD or not PTTIME */
   #ifdef PTTIME
      #ifdef NTHREADS
      omp_set_num_threads(NTHREADS);
      #endif
      #ifdef DYNAMIC 
         #pragma omp parallel for schedule(dynamic)
      #else
         #pragma omp parallel for schedule(static)
      #endif
   #endif
   for (INDEXTYPE i = 0; i < m; i++)
#endif
   {
      register VTYPE Va0, Vc0, Va1, Vc1, Va2, Vc2, Va3, Vc3, Va4, Vc4, Va5, Vc5,
                     Va6, Vc6, Va7, Vc7, Va8, Vc8, Va9, Vc9, Va10, Vc10, Va11,
                     Vc11, Va12, Vc12, Va13, Vc13, Va14, Vc14, Va15, Vc15;
      INDEXTYPE iindex = i * 128; 
      //const float *Ai = a + iindex; 
      float *Ci = c + iindex; 
      VTYPE VMAXBOUND, VMINBOUND; 
#ifdef BETA0
/*
 * NO need to load C, just zerod Vector register    
 */
      BCL_vzero(Vc0); 
      BCL_vzero(Vc1); 
      BCL_vzero(Vc2); 
      BCL_vzero(Vc3); 
      BCL_vzero(Vc4); 
      BCL_vzero(Vc5); 
      BCL_vzero(Vc6); 
      BCL_vzero(Vc7); 
      BCL_vzero(Vc8); 
      BCL_vzero(Vc9); 
      BCL_vzero(Vc10); 
      BCL_vzero(Vc11); 
      BCL_vzero(Vc12); 
      BCL_vzero(Vc13); 
      BCL_vzero(Vc14); 
      BCL_vzero(Vc15); 

#else /* beta1 */
      // load Vc 
      BCL_vldu(Vc0, Ci+VLEN*0); 
      BCL_vldu(Vc1, Ci+VLEN*1); 
      BCL_vldu(Vc2, Ci+VLEN*2); 
      BCL_vldu(Vc3, Ci+VLEN*3); 
      BCL_vldu(Vc4, Ci+VLEN*4); 
      BCL_vldu(Vc5, Ci+VLEN*5); 
      BCL_vldu(Vc6, Ci+VLEN*6); 
      BCL_vldu(Vc7, Ci+VLEN*7); 
      BCL_vldu(Vc8, Ci+VLEN*8); 
      BCL_vldu(Vc9, Ci+VLEN*9); 
      BCL_vldu(Vc10, Ci+VLEN*10); 
      BCL_vldu(Vc11, Ci+VLEN*11); 
      BCL_vldu(Vc12, Ci+VLEN*12); 
      BCL_vldu(Vc13, Ci+VLEN*13); 
      BCL_vldu(Vc14, Ci+VLEN*14); 
      BCL_vldu(Vc15, Ci+VLEN*15); 
#endif
      for (INDEXTYPE j = pntrb[i]; j < pntre[i]; j++)
      {
         VTYPE Vb0, Vb1, Vb2, Vb3, Vb4, Vb5, Vb6, Vb7, Vb8, Vb9, Vb10, Vb11,
               Vb12, Vb13, Vb14, Vb15;
         VTYPE Va0; 
         float a0 = val[j];
         INDEXTYPE colidj = indx[j];
         INDEXTYPE jindex = colidj*128;
         const float *Bj = b + jindex; 
         // load Vxj 
         BCL_vldu(Vb0, Bj+VLEN*0); 
         BCL_vldu(Vb1, Bj+VLEN*1); 
         BCL_vldu(Vb2, Bj+VLEN*2); 
         BCL_vldu(Vb3, Bj+VLEN*3); 
         BCL_vldu(Vb4, Bj+VLEN*4); 
         BCL_vldu(Vb5, Bj+VLEN*5); 
         BCL_vldu(Vb6, Bj+VLEN*6); 
         BCL_vldu(Vb7, Bj+VLEN*7); 
         BCL_vldu(Vb8, Bj+VLEN*8); 
         BCL_vldu(Vb9, Bj+VLEN*9); 
         BCL_vldu(Vb10, Bj+VLEN*10); 
         BCL_vldu(Vb11, Bj+VLEN*11); 
         BCL_vldu(Vb12, Bj+VLEN*12); 
         BCL_vldu(Vb13, Bj+VLEN*13); 
         BCL_vldu(Vb14, Bj+VLEN*14); 
         BCL_vldu(Vb15, Bj+VLEN*15); 
         BCL_vset1(Va0, a0);

         //float pf_val[8];
         //memcpy(pf_val, &Va0, sizeof(Va0));
         //printf("Va0 Numerical: %f %f %f %f %f %f %f %f \n", 
         //  pf_val[0], pf_val[1], pf_val[2], pf_val[3], pf_val[4], pf_val[5], 
         //  pf_val[6], pf_val[7]);

         // spmm vmac 
         BCL_vmac(Vc0, Va0, Vb0);

         //memcpy(pf_val, &Vc0, sizeof(Vc0));
         //printf("Vc0 Numerical: %f %f %f %f %f %f %f %f \n", 
         //  pf_val[0], pf_val[1], pf_val[2], pf_val[3], pf_val[4], pf_val[5], 
         //  pf_val[6], pf_val[7]);

         BCL_vmac(Vc1, Va0, Vb1);
         BCL_vmac(Vc2, Va0, Vb2);
         BCL_vmac(Vc3, Va0, Vb3);
         BCL_vmac(Vc4, Va0, Vb4);
         BCL_vmac(Vc5, Va0, Vb5);
         BCL_vmac(Vc6, Va0, Vb6);
         BCL_vmac(Vc7, Va0, Vb7);
         BCL_vmac(Vc8, Va0, Vb8);
         BCL_vmac(Vc9, Va0, Vb9);
         BCL_vmac(Vc10, Va0, Vb10);
         BCL_vmac(Vc11, Va0, Vb11);
         BCL_vmac(Vc12, Va0, Vb12);
         BCL_vmac(Vc13, Va0, Vb13);
         BCL_vmac(Vc14, Va0, Vb14);
         BCL_vmac(Vc15, Va0, Vb15);
      }
      BCL_vstu(Ci + VLEN*0, Vc0); 
      BCL_vstu(Ci + VLEN*1, Vc1); 
      BCL_vstu(Ci + VLEN*2, Vc2); 
      BCL_vstu(Ci + VLEN*3, Vc3); 
      BCL_vstu(Ci + VLEN*4, Vc4); 
      BCL_vstu(Ci + VLEN*5, Vc5); 
      BCL_vstu(Ci + VLEN*6, Vc6); 
      BCL_vstu(Ci + VLEN*7, Vc7); 
      BCL_vstu(Ci + VLEN*8, Vc8); 
      BCL_vstu(Ci + VLEN*9, Vc9); 
      BCL_vstu(Ci + VLEN*10, Vc10); 
      BCL_vstu(Ci + VLEN*11, Vc11); 
      BCL_vstu(Ci + VLEN*12, Vc12); 
      BCL_vstu(Ci + VLEN*13, Vc13); 
      BCL_vstu(Ci + VLEN*14, Vc14); 
      BCL_vstu(Ci + VLEN*15, Vc15); 
   }
#if defined(PTTIME) && defined(LDB)
   }
#endif
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

  //printf("pntrb[0] = %lu\n" ,pntrb[0]);
  //printf("pntrb[1] = %lu\n" ,pntrb[1]);
  //printf("pntrb[2] = %lu\n" ,pntrb[2]);
  //printf("pntrb[3] = %lu\n" ,pntrb[3]);

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

  //printf("pntre[0] = %lu\n" ,pntre[0]);
  //printf("pntre[1] = %lu\n" ,pntre[1]);
  //printf("pntre[2] = %lu\n" ,pntre[2]);
  //printf("pntre[3] = %lu\n" ,pntre[3]);

  // b alloc
  VALUETYPE* b = (VALUETYPE*) aligned_alloc(CACHE_LINE_SIZE,  num_node*dim_vector_hvd * sizeof(VALUETYPE));
  INDEXTYPE ldb = dim_vector_hvd;

  // b from file
  FILE* fp_mtx4 = fopen("../dataset/graph/harvard/harvard_b_mat.dat", "rb");

  if (fp_mtx4 != NULL) {
    fseek(fp_mtx4, 0L, SEEK_END);
    uint64_t sz = ftell(fp_mtx4);
    fseek(fp_mtx4, 0L, SEEK_SET);
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

  for (uint64_t i = 0; i < num_iter; i++) {
	  sgfusedMM_K128_spmm_b0_csr(m, k, val, indx, pntrb, pntre, b, ldb, c, ldc);
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
