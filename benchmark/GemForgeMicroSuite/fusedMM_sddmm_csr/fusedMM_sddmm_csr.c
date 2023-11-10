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
#include <math.h>

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
static const float sm_bound = 5.0;
static const int sm_table_size = 2048;
static const float sm_resolution = sm_table_size/(2.0 * sm_bound);

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

#include<stdint.h>
#ifdef PTTIME
   #include<omp.h>
#endif
#define SREAL 1
#include "simd.h"
#if VLEN != 8
   #error "ARCH VLEN doesn't match with generator's VLEN, see simd.h " 
#endif


// NOT WORKING
// SOP_INHOUSE

#define SIGMOID_UDEF
#define BETA0

/*
 * NOTE: The implementation of User defined functions differ from different 
 * models. We need to enable/disable it compile time!!!!
 */

VALUETYPE ufast_SM(VALUETYPE v, VALUETYPE *sm_table)
{  
   int sm_idx = (v + sm_bound) * sm_resolution;

   if (sm_idx >= sm_table_size ) sm_idx = sm_table_size-1;
   else if (sm_idx < 0)          sm_idx = 0;
   return sm_table[sm_idx];
}

#define FUSEDMM_SUCCESS_RETURN 0
#ifdef SIGMOID_UDEF 
// USER DEFINED FUNCTION for SOP with Sigmoid calc 
//int  SOP_UDEF_FUNC(VALUETYPE val, VALUETYPE *out)
int  SOP_UDEF_FUNC(VALUETYPE val, VALUETYPE *sm_table, VALUETYPE *out)
{
   *out = 1.0 - ufast_SM(val, sm_table);
   return FUSEDMM_SUCCESS_RETURN;
}
#elif defined(FR_UDEF)
// SOP_UDEF for FR model
int SOP_UDEF_FUNC(VALUETYPE val, VALUETYPE *out)
{
   *out = 1.0 + 1.0 / val;
   return FUSEDMM_SUCCESS_RETURN;
}
#elif defined(TDIST_UDEF)
// SOP_UDEF for t-distribution  model
int SOP_UDEF_FUNC(VALUETYPE val, VALUETYPE *out)
{  
   *out = tscale(-2.0 / (1.0 + val));
   return FUSEDMM_SUCCESS_RETURN;
}
#elif defined(LL_UDEF)
// SOP_UDEF for LL model
int SOP_UDEF_FUNC(VALUETYPE val, VALUETYPE *out)
{
   *out = log2(1 + sqrt(val));;
   return FUSEDMM_SUCCESS_RETURN;
}
#elif defined(FA_UDEF)
// SOP_UDEF for FA model
int SOP_UDEF_FUNC(VALUETYPE val, VALUETYPE *out)
{
   *out = sqrt(val) + 1.0 / val;;
   return FUSEDMM_SUCCESS_RETURN;
}
#else 
/*
 * NOTE: other kernels don't use SOP funciton (NOOP or COPY)
 * However, since we enable SOP_UDEF_IMPL in fusedMM.h, we need a dummy func.
 * Normally, users should disable the macro if they don't want to provide any 
 * implementation. We are using this dummy since we use same source for all 
 * the different executables.
 */
int SOP_UDEF_FUNC(VALUETYPE val, VALUETYPE *out)
{
   *out = val;;
   return FUSEDMM_SUCCESS_RETURN;
} 
#endif

/*
 * Register block  A,C and B(innermost loop) will require most registers, works 
 * better on small value of k
 */
//#ifndef SOP_INHOUSE 
//extern int SOP_UDEF_FUNC(float val, float *out);  
//#endif
/* external declaration of misc functions  */
#ifdef BETA0 
//void sgfusedMM_K128_sigmoid_b0_csr
__attribute__((noinline)) Value sgfusedMM_K128_sigmoid_b0_csr
#else /* BETA1 version */
//void sgfusedMM_K128_sigmoid_b1_csr
__attribute__((noinline)) Value sgfusedMM_K128_sigmoid_b1_csr
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
   const float *a,        // Dense A matrix
   const INDEXTYPE lda,    // leading dimension of a (col size since row-major)  
   const float *b,        // Dense B matrix
   const INDEXTYPE ldb,    // leading dimension of b (col size since row-major)  
//   const float beta,      // beta value, compile time not used  
   float *c,              // Dense matrix c
   const INDEXTYPE ldc,     // leading dimension size of c (col size since roa-major) 
   float *user_sm_table
)
{


#ifdef SOP_INHOUSE  
   float *sm_table = (float*)malloc(sizeof(float)*sm_table_size);   
   if (!sm_table)
   {
      fprintf(stderr, 
      "Not enough memory to allocate SM TABLE in kernel, SM_TABLE_SIZE = %d!!!\n", 
              sm_table_size);
      exit(0);
   }
   { // init_sm_table 
      for(INDEXTYPE i = 0; i < sm_table_size; i++)
      {
         float x;
         x = 2.0 * sm_bound * i / sm_table_size - sm_bound;
         sm_table[i] = 1.0 / (1 + exp(-x));
      }
   }
#endif

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
      const float *Ai = a + iindex; 
      float *Ci = c + iindex; 
      VTYPE VMAXBOUND, VMINBOUND; 
#ifdef SOP_INHOUSE
      BCL_vset1(VMAXBOUND, sm_bound); 
      BCL_vset1(VMINBOUND, -sm_bound); 
#endif
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
      // load Va 
      BCL_vldu(Va0, Ai+VLEN*0); 
      BCL_vldu(Va1, Ai+VLEN*1); 
      BCL_vldu(Va2, Ai+VLEN*2); 
      BCL_vldu(Va3, Ai+VLEN*3); 
      BCL_vldu(Va4, Ai+VLEN*4); 
      BCL_vldu(Va5, Ai+VLEN*5); 
      BCL_vldu(Va6, Ai+VLEN*6); 
      BCL_vldu(Va7, Ai+VLEN*7); 
      BCL_vldu(Va8, Ai+VLEN*8); 
      BCL_vldu(Va9, Ai+VLEN*9); 
      BCL_vldu(Va10, Ai+VLEN*10); 
      BCL_vldu(Va11, Ai+VLEN*11); 
      BCL_vldu(Va12, Ai+VLEN*12); 
      BCL_vldu(Va13, Ai+VLEN*13); 
      BCL_vldu(Va14, Ai+VLEN*14); 
      BCL_vldu(Va15, Ai+VLEN*15); 
	  
      for (INDEXTYPE j = pntrb[i]; j < pntre[i]; j++)
      {
         VTYPE Vb0, Vb1, Vb2, Vb3, Vb4, Vb5, Vb6, Vb7, Vb8, Vb9, Vb10, Vb11,
               Vb12, Vb13, Vb14, Vb15;
         VTYPE Vd0, Vd1; 
         float d1;
         VTYPE Vatt0, Vatt1, Vatt2, Vatt3, Vatt4, Vatt5, Vatt6, Vatt7, Vatt8,
               Vatt9, Vatt10, Vatt11, Vatt12, Vatt13, Vatt14, Vatt15;
         //float attrc = 0;
         VALUETYPE attrc = 0;
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
      // init Vatt  
         BCL_vzero(Vatt0);
         BCL_vzero(Vatt1);
         BCL_vzero(Vatt2);
         BCL_vzero(Vatt3);
         BCL_vzero(Vatt4);
         BCL_vzero(Vatt5);
         BCL_vzero(Vatt6);
         BCL_vzero(Vatt7);
         BCL_vzero(Vatt8);
         BCL_vzero(Vatt9);
         BCL_vzero(Vatt10);
         BCL_vzero(Vatt11);
         BCL_vzero(Vatt12);
         BCL_vzero(Vatt13);
         BCL_vzero(Vatt14);
         BCL_vzero(Vatt15);
		 
      // vmac 
         BCL_vmac(Vatt0, Va0, Vb0);
         BCL_vmac(Vatt1, Va1, Vb1);
         BCL_vmac(Vatt2, Va2, Vb2);
         BCL_vmac(Vatt3, Va3, Vb3);
         BCL_vmac(Vatt4, Va4, Vb4);
         BCL_vmac(Vatt5, Va5, Vb5);
         BCL_vmac(Vatt6, Va6, Vb6);
         BCL_vmac(Vatt7, Va7, Vb7);
         BCL_vmac(Vatt8, Va8, Vb8);
         BCL_vmac(Vatt9, Va9, Vb9);
         BCL_vmac(Vatt10, Va10, Vb10);
         BCL_vmac(Vatt11, Va11, Vb11);
         BCL_vmac(Vatt12, Va12, Vb12);
         BCL_vmac(Vatt13, Va13, Vb13);
         BCL_vmac(Vatt14, Va14, Vb14);
         BCL_vmac(Vatt15, Va15, Vb15);
         // binary tree reduction 
         BCL_vadd(Vatt0, Vatt0, Vatt1);
         BCL_vadd(Vatt2, Vatt2, Vatt3);
         BCL_vadd(Vatt4, Vatt4, Vatt5);
         BCL_vadd(Vatt6, Vatt6, Vatt7);
         BCL_vadd(Vatt8, Vatt8, Vatt9);
         BCL_vadd(Vatt10, Vatt10, Vatt11);
         BCL_vadd(Vatt12, Vatt12, Vatt13);
         BCL_vadd(Vatt14, Vatt14, Vatt15);
         BCL_vadd(Vatt0, Vatt0, Vatt2);
         BCL_vadd(Vatt4, Vatt4, Vatt6);
         BCL_vadd(Vatt8, Vatt8, Vatt10);
         BCL_vadd(Vatt12, Vatt12, Vatt14);
         BCL_vadd(Vatt0, Vatt0, Vatt4);
         BCL_vadd(Vatt8, Vatt8, Vatt12);
         BCL_vadd(Vatt0, Vatt0, Vatt8);
         
		 //float pf_val[8];
         //memcpy(pf_val, &Vatt0, sizeof(Vatt0));
         //printf("Vatt0 Numerical: %f %f %f %f %f %f %f %f \n", 
         //  pf_val[0], pf_val[1], pf_val[2], pf_val[3], pf_val[4], pf_val[5], 
         //  pf_val[6], pf_val[7]);

//      BCL_vrsum1(attrc, Vatt0);
		BCL_vrsum_syo(attrc, Vatt0);

#ifdef SOP_INHOUSE
         /* Calculating Sigmoid value */
         { // fast_SM 
            //d1 = fast_SM(attrc, sm_table);
            if (attrc > sm_bound) d1 = 1.0;
            else if (attrc < -sm_bound) d1 = 0.0;
            else d1 = sm_table[(INDEXTYPE) ((attrc+sm_bound)*sm_resolution)];
         }
         //d1 = STEP * degi * (1.0 - d1);
         d1 = (1.0 - d1);
#else		 
         //SOP_UDEF_FUNC(attrc, &d1);
         SOP_UDEF_FUNC(attrc, user_sm_table, &d1);
#endif
         BCL_vset1(Vd1, d1);

         // vmac 
         BCL_vmac(Vc0, Vd1, Vb0);
         BCL_vmac(Vc1, Vd1, Vb1);
         BCL_vmac(Vc2, Vd1, Vb2);
         BCL_vmac(Vc3, Vd1, Vb3);
         BCL_vmac(Vc4, Vd1, Vb4);
         BCL_vmac(Vc5, Vd1, Vb5);
         BCL_vmac(Vc6, Vd1, Vb6);
         BCL_vmac(Vc7, Vd1, Vb7);
         BCL_vmac(Vc8, Vd1, Vb8);
         BCL_vmac(Vc9, Vd1, Vb9);
         BCL_vmac(Vc10, Vd1, Vb10);
         BCL_vmac(Vc11, Vd1, Vb11);
         BCL_vmac(Vc12, Vd1, Vb12);
         BCL_vmac(Vc13, Vd1, Vb13);
         BCL_vmac(Vc14, Vd1, Vb14);
         BCL_vmac(Vc15, Vd1, Vb15);
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
#ifdef SOP_INHOUSE
   free(sm_table);
#endif
   // yosong, 231103
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

  // a alloc
  VALUETYPE* a = (VALUETYPE*) aligned_alloc(CACHE_LINE_SIZE,  num_node*dim_vector_hvd * sizeof(VALUETYPE));
  INDEXTYPE lda = dim_vector_hvd;

  // b from file
  FILE* fp_mtx6 = fopen("../dataset/graph/harvard/harvard_a_mat.dat", "rb");

  if (fp_mtx6 != NULL) {
    fseek(fp_mtx6, 0L, SEEK_END);
    uint64_t sz = ftell(fp_mtx6);
    fseek(fp_mtx6, 0L, SEEK_SET);
    if (sz == num_node*dim_vector_hvd * sizeof(VALUETYPE)) {
      fread((void*)a, sizeof(VALUETYPE), num_node*dim_vector_hvd, fp_mtx6);
    }
    fclose(fp_mtx6);	
  }
  else {
    printf("Cannot find harvard_a_mat.dat\n");
    return 0;
  }

  //printf("a[0] = %f\n" ,a[0]);
  //printf("a[1] = %f\n" ,a[1]);
  //printf("a[2] = %f\n" ,a[2]);
  //printf("a[3] = %f\n" ,a[3]);

  VALUETYPE* sm_table = (VALUETYPE*) aligned_alloc(CACHE_LINE_SIZE,  sm_table_size * sizeof(VALUETYPE));
  FILE* fp_mtx7 = fopen("../dataset/sigmoid/sm_table.dat", "rb");

  if (fp_mtx7 != NULL) {
    fseek(fp_mtx7, 0L, SEEK_END);
    uint64_t sz = ftell(fp_mtx7);
    fseek(fp_mtx7, 0L, SEEK_SET);
    if (sz == sm_table_size * sizeof(VALUETYPE)) {
      fread((void*)sm_table, sizeof(VALUETYPE),sm_table_size, fp_mtx7);
    }
    fclose(fp_mtx7);
  }
  else {
    printf("Cannot find sm_table.dat\n");
    return 0;
  }

  //printf("sm_table[0] = %f\n" , sm_table[0]);
  //printf("sm_table[1] = %f\n" , sm_table[1]);
  //printf("sm_table[2] = %f\n" , sm_table[2]);
  //printf("sm_table[3] = %f\n" , sm_table[3]);


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
      //truested_spmm_csr(m, k, val, indx, pntrb, pntre, b, ldb, c, ldc);
      sgfusedMM_K128_sigmoid_b0_csr(m, k, val, indx, pntrb, pntre, a, lda, b, ldb, c, ldc, sm_table);
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
  free(a);
  free(b);
  free(c);
  free(sm_table);
  return 0;
}
