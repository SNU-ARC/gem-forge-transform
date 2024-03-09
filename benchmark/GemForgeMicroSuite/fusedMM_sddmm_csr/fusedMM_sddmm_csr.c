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
static const uint64_t num_iter			  = 1;

//#define CHECK

//#define PSP

typedef float Value;
static const float sm_bound = 5.0;
static const int sm_table_size = 2048;
static const float sm_resolution = sm_table_size/(2.0 * sm_bound);

// Turn off Cache warm-up by default, because it is not our assumption
//#define WARM_CACHE

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

#if DIM == 104
#ifdef BETA0 
__attribute__((noinline)) Value sgfusedMM_K104_sigmoid_b0_csr
#else /* BETA1 version */
__attribute__((noinline)) Value sgfusedMM_K104_sigmoid_b1_csr
#endif
(
   const INDEXTYPE m,      // rows of dense A matrix 
   const INDEXTYPE k,      // cols of A or dimension. not used since K compile time   
   const float *val,       // value of  the sparse matrix 
   const INDEXTYPE *indx,  // colids -> column indices of sparse matrix 
   const INDEXTYPE *pntrb, // starting index for rowptr of csr of sparse matrix
   const INDEXTYPE *pntre, // ending index for rowptr of csr of sparse matrix 
   const float *a,        // Dense A matrix
   const INDEXTYPE lda,    // leading dimension of a (col size since row-major)  
   const float *b,        // Dense B matrix
   const INDEXTYPE ldb,    // leading dimension of b (col size since row-major)  
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
                     Vc11, Va12, Vc12;
      INDEXTYPE iindex = i * 104; 
      const float *Ai = a + iindex; 
      float *Ci = c + iindex; 
      VTYPE VMAXBOUND, VMINBOUND; 
#ifdef SOP_INHOUSE
      BCL_vset1(VMAXBOUND, sm_bound); 
      BCL_vset1(VMINBOUND, -sm_bound); 
#endif
     INDEXTYPE offset_begin = pntrb[i];
     INDEXTYPE offset_end = pntre[i];
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

      for (INDEXTYPE j = pntrb[i]; j < pntre[i]; j++)
      {
         VTYPE Vb0, Vb1, Vb2, Vb3, Vb4, Vb5, Vb6, Vb7, Vb8, Vb9, Vb10, Vb11,
               Vb12;
         VTYPE Vd0, Vd1; 
         float d1;
         VTYPE Vatt0, Vatt1, Vatt2, Vatt3, Vatt4, Vatt5, Vatt6, Vatt7, Vatt8,
               Vatt9, Vatt10, Vatt11, Vatt12;
         float attrc = 0;
         INDEXTYPE colidj = indx[j];
         INDEXTYPE jindex = colidj*104;
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
         // binary tree reduction 
         BCL_vadd(Vatt0, Vatt0, Vatt1);
         BCL_vadd(Vatt2, Vatt2, Vatt3);
         BCL_vadd(Vatt4, Vatt4, Vatt5);
         BCL_vadd(Vatt6, Vatt6, Vatt7);
         BCL_vadd(Vatt8, Vatt8, Vatt9);
         BCL_vadd(Vatt10, Vatt10, Vatt11);
         BCL_vadd(Vatt0, Vatt0, Vatt2);
         BCL_vadd(Vatt4, Vatt4, Vatt6);
         BCL_vadd(Vatt8, Vatt8, Vatt10);
         BCL_vadd(Vatt0, Vatt0, Vatt4);
         BCL_vadd(Vatt8, Vatt8, Vatt12);
         BCL_vadd(Vatt0, Vatt0, Vatt8);

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
   }
#if defined(PTTIME) && defined(LDB)
   }
#endif
#ifdef SOP_INHOUSE
   free(sm_table);
#endif
   return 0;
}
#elif DIM == 128
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
     INDEXTYPE offset_begin = pntrb[i];
     INDEXTYPE offset_end = pntre[i];
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
#elif DIM == 256
#ifdef BETA0 
__attribute__((noinline)) Value sgfusedMM_K256_sigmoid_b0_csr
#else /* BETA1 version */
__attribute__((noinline)) Value sgfusedMM_K256_sigmoid_b1_csr
#endif
(
   const INDEXTYPE m,      // rows of dense A matrix 
   const INDEXTYPE k,      // cols of A or dimension. not used since K compile time   
   const float *val,       // value of  the sparse matrix 
   const INDEXTYPE *indx,  // colids -> column indices of sparse matrix 
   const INDEXTYPE *pntrb, // starting index for rowptr of csr of sparse matrix
   const INDEXTYPE *pntre, // ending index for rowptr of csr of sparse matrix 
   const float *a,        // Dense A matrix
   const INDEXTYPE lda,    // leading dimension of a (col size since row-major)  
   const float *b,        // Dense B matrix
   const INDEXTYPE ldb,    // leading dimension of b (col size since row-major)  
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
                     Vc11, Va12, Vc12, Va13, Vc13, Va14, Vc14, Va15, Vc15, Va16,
                     Vc16, Va17, Vc17, Va18, Vc18, Va19, Vc19, Va20, Vc20, Va21,
                     Vc21, Va22, Vc22, Va23, Vc23, Va24, Vc24, Va25, Vc25, Va26,
                     Vc26, Va27, Vc27, Va28, Vc28, Va29, Vc29, Va30, Vc30, Va31,
                     Vc31;
      INDEXTYPE iindex = i * 256; 
      const float *Ai = a + iindex; 
      float *Ci = c + iindex; 
      VTYPE VMAXBOUND, VMINBOUND; 
#ifdef SOP_INHOUSE
      BCL_vset1(VMAXBOUND, sm_bound); 
      BCL_vset1(VMINBOUND, -sm_bound); 
#endif
     INDEXTYPE offset_begin = pntrb[i];
     INDEXTYPE offset_end = pntre[i];
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
      BCL_vzero(Vc16); 
      BCL_vzero(Vc17); 
      BCL_vzero(Vc18); 
      BCL_vzero(Vc19); 
      BCL_vzero(Vc20); 
      BCL_vzero(Vc21); 
      BCL_vzero(Vc22); 
      BCL_vzero(Vc23); 
      BCL_vzero(Vc24); 
      BCL_vzero(Vc25); 
      BCL_vzero(Vc26); 
      BCL_vzero(Vc27); 
      BCL_vzero(Vc28); 
      BCL_vzero(Vc29); 
      BCL_vzero(Vc30); 
      BCL_vzero(Vc31); 

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
      BCL_vldu(Vc16, Ci+VLEN*16); 
      BCL_vldu(Vc17, Ci+VLEN*17); 
      BCL_vldu(Vc18, Ci+VLEN*18); 
      BCL_vldu(Vc19, Ci+VLEN*19); 
      BCL_vldu(Vc20, Ci+VLEN*20); 
      BCL_vldu(Vc21, Ci+VLEN*21); 
      BCL_vldu(Vc22, Ci+VLEN*22); 
      BCL_vldu(Vc23, Ci+VLEN*23); 
      BCL_vldu(Vc24, Ci+VLEN*24); 
      BCL_vldu(Vc25, Ci+VLEN*25); 
      BCL_vldu(Vc26, Ci+VLEN*26); 
      BCL_vldu(Vc27, Ci+VLEN*27); 
      BCL_vldu(Vc28, Ci+VLEN*28); 
      BCL_vldu(Vc29, Ci+VLEN*29); 
      BCL_vldu(Vc30, Ci+VLEN*30); 
      BCL_vldu(Vc31, Ci+VLEN*31); 
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
      BCL_vldu(Va16, Ai+VLEN*16); 
      BCL_vldu(Va17, Ai+VLEN*17); 
      BCL_vldu(Va18, Ai+VLEN*18); 
      BCL_vldu(Va19, Ai+VLEN*19); 
      BCL_vldu(Va20, Ai+VLEN*20); 
      BCL_vldu(Va21, Ai+VLEN*21); 
      BCL_vldu(Va22, Ai+VLEN*22); 
      BCL_vldu(Va23, Ai+VLEN*23); 
      BCL_vldu(Va24, Ai+VLEN*24); 
      BCL_vldu(Va25, Ai+VLEN*25); 
      BCL_vldu(Va26, Ai+VLEN*26); 
      BCL_vldu(Va27, Ai+VLEN*27); 
      BCL_vldu(Va28, Ai+VLEN*28); 
      BCL_vldu(Va29, Ai+VLEN*29); 
      BCL_vldu(Va30, Ai+VLEN*30); 
      BCL_vldu(Va31, Ai+VLEN*31); 

      for (INDEXTYPE j = pntrb[i]; j < pntre[i]; j++)
      {
         VTYPE Vb0, Vb1, Vb2, Vb3, Vb4, Vb5, Vb6, Vb7, Vb8, Vb9, Vb10, Vb11,
               Vb12, Vb13, Vb14, Vb15, Vb16, Vb17, Vb18, Vb19, Vb20, Vb21, Vb22,
               Vb23, Vb24, Vb25, Vb26, Vb27, Vb28, Vb29, Vb30, Vb31;
         VTYPE Vd0, Vd1; 
         float d1;
         VTYPE Vatt0, Vatt1, Vatt2, Vatt3, Vatt4, Vatt5, Vatt6, Vatt7, Vatt8,
               Vatt9, Vatt10, Vatt11, Vatt12, Vatt13, Vatt14, Vatt15, Vatt16,
               Vatt17, Vatt18, Vatt19, Vatt20, Vatt21, Vatt22, Vatt23, Vatt24,
               Vatt25, Vatt26, Vatt27, Vatt28, Vatt29, Vatt30, Vatt31;
         float attrc = 0;
         INDEXTYPE colidj = indx[j];
         INDEXTYPE jindex = colidj*256;
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
         BCL_vldu(Vb16, Bj+VLEN*16); 
         BCL_vldu(Vb17, Bj+VLEN*17); 
         BCL_vldu(Vb18, Bj+VLEN*18); 
         BCL_vldu(Vb19, Bj+VLEN*19); 
         BCL_vldu(Vb20, Bj+VLEN*20); 
         BCL_vldu(Vb21, Bj+VLEN*21); 
         BCL_vldu(Vb22, Bj+VLEN*22); 
         BCL_vldu(Vb23, Bj+VLEN*23); 
         BCL_vldu(Vb24, Bj+VLEN*24); 
         BCL_vldu(Vb25, Bj+VLEN*25); 
         BCL_vldu(Vb26, Bj+VLEN*26); 
         BCL_vldu(Vb27, Bj+VLEN*27); 
         BCL_vldu(Vb28, Bj+VLEN*28); 
         BCL_vldu(Vb29, Bj+VLEN*29); 
         BCL_vldu(Vb30, Bj+VLEN*30); 
         BCL_vldu(Vb31, Bj+VLEN*31); 
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
         BCL_vzero(Vatt16);
         BCL_vzero(Vatt17);
         BCL_vzero(Vatt18);
         BCL_vzero(Vatt19);
         BCL_vzero(Vatt20);
         BCL_vzero(Vatt21);
         BCL_vzero(Vatt22);
         BCL_vzero(Vatt23);
         BCL_vzero(Vatt24);
         BCL_vzero(Vatt25);
         BCL_vzero(Vatt26);
         BCL_vzero(Vatt27);
         BCL_vzero(Vatt28);
         BCL_vzero(Vatt29);
         BCL_vzero(Vatt30);
         BCL_vzero(Vatt31);

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
         BCL_vmac(Vatt16, Va16, Vb16);
         BCL_vmac(Vatt17, Va17, Vb17);
         BCL_vmac(Vatt18, Va18, Vb18);
         BCL_vmac(Vatt19, Va19, Vb19);
         BCL_vmac(Vatt20, Va20, Vb20);
         BCL_vmac(Vatt21, Va21, Vb21);
         BCL_vmac(Vatt22, Va22, Vb22);
         BCL_vmac(Vatt23, Va23, Vb23);
         BCL_vmac(Vatt24, Va24, Vb24);
         BCL_vmac(Vatt25, Va25, Vb25);
         BCL_vmac(Vatt26, Va26, Vb26);
         BCL_vmac(Vatt27, Va27, Vb27);
         BCL_vmac(Vatt28, Va28, Vb28);
         BCL_vmac(Vatt29, Va29, Vb29);
         BCL_vmac(Vatt30, Va30, Vb30);
         BCL_vmac(Vatt31, Va31, Vb31);
         // binary tree reduction 
         BCL_vadd(Vatt0, Vatt0, Vatt1);
         BCL_vadd(Vatt2, Vatt2, Vatt3);
         BCL_vadd(Vatt4, Vatt4, Vatt5);
         BCL_vadd(Vatt6, Vatt6, Vatt7);
         BCL_vadd(Vatt8, Vatt8, Vatt9);
         BCL_vadd(Vatt10, Vatt10, Vatt11);
         BCL_vadd(Vatt12, Vatt12, Vatt13);
         BCL_vadd(Vatt14, Vatt14, Vatt15);
         BCL_vadd(Vatt16, Vatt16, Vatt17);
         BCL_vadd(Vatt18, Vatt18, Vatt19);
         BCL_vadd(Vatt20, Vatt20, Vatt21);
         BCL_vadd(Vatt22, Vatt22, Vatt23);
         BCL_vadd(Vatt24, Vatt24, Vatt25);
         BCL_vadd(Vatt26, Vatt26, Vatt27);
         BCL_vadd(Vatt28, Vatt28, Vatt29);
         BCL_vadd(Vatt30, Vatt30, Vatt31);
         BCL_vadd(Vatt0, Vatt0, Vatt2);
         BCL_vadd(Vatt4, Vatt4, Vatt6);
         BCL_vadd(Vatt8, Vatt8, Vatt10);
         BCL_vadd(Vatt12, Vatt12, Vatt14);
         BCL_vadd(Vatt16, Vatt16, Vatt18);
         BCL_vadd(Vatt20, Vatt20, Vatt22);
         BCL_vadd(Vatt24, Vatt24, Vatt26);
         BCL_vadd(Vatt28, Vatt28, Vatt30);
         BCL_vadd(Vatt0, Vatt0, Vatt4);
         BCL_vadd(Vatt8, Vatt8, Vatt12);
         BCL_vadd(Vatt16, Vatt16, Vatt20);
         BCL_vadd(Vatt24, Vatt24, Vatt28);
         BCL_vadd(Vatt0, Vatt0, Vatt8);
         BCL_vadd(Vatt16, Vatt16, Vatt24);
         BCL_vadd(Vatt0, Vatt0, Vatt16);

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
         BCL_vmac(Vc16, Vd1, Vb16);
         BCL_vmac(Vc17, Vd1, Vb17);
         BCL_vmac(Vc18, Vd1, Vb18);
         BCL_vmac(Vc19, Vd1, Vb19);
         BCL_vmac(Vc20, Vd1, Vb20);
         BCL_vmac(Vc21, Vd1, Vb21);
         BCL_vmac(Vc22, Vd1, Vb22);
         BCL_vmac(Vc23, Vd1, Vb23);
         BCL_vmac(Vc24, Vd1, Vb24);
         BCL_vmac(Vc25, Vd1, Vb25);
         BCL_vmac(Vc26, Vd1, Vb26);
         BCL_vmac(Vc27, Vd1, Vb27);
         BCL_vmac(Vc28, Vd1, Vb28);
         BCL_vmac(Vc29, Vd1, Vb29);
         BCL_vmac(Vc30, Vd1, Vb30);
         BCL_vmac(Vc31, Vd1, Vb31);
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
      BCL_vstu(Ci + VLEN*16, Vc16); 
      BCL_vstu(Ci + VLEN*17, Vc17); 
      BCL_vstu(Ci + VLEN*18, Vc18); 
      BCL_vstu(Ci + VLEN*19, Vc19); 
      BCL_vstu(Ci + VLEN*20, Vc20); 
      BCL_vstu(Ci + VLEN*21, Vc21); 
      BCL_vstu(Ci + VLEN*22, Vc22); 
      BCL_vstu(Ci + VLEN*23, Vc23); 
      BCL_vstu(Ci + VLEN*24, Vc24); 
      BCL_vstu(Ci + VLEN*25, Vc25); 
      BCL_vstu(Ci + VLEN*26, Vc26); 
      BCL_vstu(Ci + VLEN*27, Vc27); 
      BCL_vstu(Ci + VLEN*28, Vc28); 
      BCL_vstu(Ci + VLEN*29, Vc29); 
      BCL_vstu(Ci + VLEN*30, Vc30); 
      BCL_vstu(Ci + VLEN*31, Vc31); 
   }
#if defined(PTTIME) && defined(LDB)
   }
#endif
#ifdef SOP_INHOUSE
   free(sm_table);
#endif
   return 0;
}
#elif DIM == 304
#ifdef BETA0 
__attribute__((noinline)) Value sgfusedMM_K304_sigmoid_b0_csr
#else /* BETA1 version */
__attribute__((noinline)) Value sgfusedMM_K304_sigmoid_b1_csr
#endif
(
   const INDEXTYPE m,      // rows of dense A matrix 
   const INDEXTYPE k,      // cols of A or dimension. not used since K compile time   
   const float *val,       // value of  the sparse matrix 
   const INDEXTYPE *indx,  // colids -> column indices of sparse matrix 
   const INDEXTYPE *pntrb, // starting index for rowptr of csr of sparse matrix
   const INDEXTYPE *pntre, // ending index for rowptr of csr of sparse matrix 
   const float *a,        // Dense A matrix
   const INDEXTYPE lda,    // leading dimension of a (col size since row-major)  
   const float *b,        // Dense B matrix
   const INDEXTYPE ldb,    // leading dimension of b (col size since row-major)  
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
                     Vc11, Va12, Vc12, Va13, Vc13, Va14, Vc14, Va15, Vc15, Va16,
                     Vc16, Va17, Vc17, Va18, Vc18, Va19, Vc19, Va20, Vc20, Va21,
                     Vc21, Va22, Vc22, Va23, Vc23, Va24, Vc24, Va25, Vc25, Va26,
                     Vc26, Va27, Vc27, Va28, Vc28, Va29, Vc29, Va30, Vc30, Va31,
                     Vc31, Va32, Vc32, Va33, Vc33, Va34, Vc34, Va35, Vc35, Va36,
                     Vc36, Va37, Vc37;
      INDEXTYPE iindex = i * 304; 
      const float *Ai = a + iindex; 
      float *Ci = c + iindex; 
      VTYPE VMAXBOUND, VMINBOUND; 
#ifdef SOP_INHOUSE
      BCL_vset1(VMAXBOUND, sm_bound); 
      BCL_vset1(VMINBOUND, -sm_bound); 
#endif
     INDEXTYPE offset_begin = pntrb[i];
     INDEXTYPE offset_end = pntre[i];
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
      BCL_vzero(Vc16); 
      BCL_vzero(Vc17); 
      BCL_vzero(Vc18); 
      BCL_vzero(Vc19); 
      BCL_vzero(Vc20); 
      BCL_vzero(Vc21); 
      BCL_vzero(Vc22); 
      BCL_vzero(Vc23); 
      BCL_vzero(Vc24); 
      BCL_vzero(Vc25); 
      BCL_vzero(Vc26); 
      BCL_vzero(Vc27); 
      BCL_vzero(Vc28); 
      BCL_vzero(Vc29); 
      BCL_vzero(Vc30); 
      BCL_vzero(Vc31); 
      BCL_vzero(Vc32); 
      BCL_vzero(Vc33); 
      BCL_vzero(Vc34); 
      BCL_vzero(Vc35); 
      BCL_vzero(Vc36); 
      BCL_vzero(Vc37); 

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
      BCL_vldu(Vc16, Ci+VLEN*16); 
      BCL_vldu(Vc17, Ci+VLEN*17); 
      BCL_vldu(Vc18, Ci+VLEN*18); 
      BCL_vldu(Vc19, Ci+VLEN*19); 
      BCL_vldu(Vc20, Ci+VLEN*20); 
      BCL_vldu(Vc21, Ci+VLEN*21); 
      BCL_vldu(Vc22, Ci+VLEN*22); 
      BCL_vldu(Vc23, Ci+VLEN*23); 
      BCL_vldu(Vc24, Ci+VLEN*24); 
      BCL_vldu(Vc25, Ci+VLEN*25); 
      BCL_vldu(Vc26, Ci+VLEN*26); 
      BCL_vldu(Vc27, Ci+VLEN*27); 
      BCL_vldu(Vc28, Ci+VLEN*28); 
      BCL_vldu(Vc29, Ci+VLEN*29); 
      BCL_vldu(Vc30, Ci+VLEN*30); 
      BCL_vldu(Vc31, Ci+VLEN*31); 
      BCL_vldu(Vc32, Ci+VLEN*32); 
      BCL_vldu(Vc33, Ci+VLEN*33); 
      BCL_vldu(Vc34, Ci+VLEN*34); 
      BCL_vldu(Vc35, Ci+VLEN*35); 
      BCL_vldu(Vc36, Ci+VLEN*36); 
      BCL_vldu(Vc37, Ci+VLEN*37); 
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
      BCL_vldu(Va16, Ai+VLEN*16); 
      BCL_vldu(Va17, Ai+VLEN*17); 
      BCL_vldu(Va18, Ai+VLEN*18); 
      BCL_vldu(Va19, Ai+VLEN*19); 
      BCL_vldu(Va20, Ai+VLEN*20); 
      BCL_vldu(Va21, Ai+VLEN*21); 
      BCL_vldu(Va22, Ai+VLEN*22); 
      BCL_vldu(Va23, Ai+VLEN*23); 
      BCL_vldu(Va24, Ai+VLEN*24); 
      BCL_vldu(Va25, Ai+VLEN*25); 
      BCL_vldu(Va26, Ai+VLEN*26); 
      BCL_vldu(Va27, Ai+VLEN*27); 
      BCL_vldu(Va28, Ai+VLEN*28); 
      BCL_vldu(Va29, Ai+VLEN*29); 
      BCL_vldu(Va30, Ai+VLEN*30); 
      BCL_vldu(Va31, Ai+VLEN*31); 
      BCL_vldu(Va32, Ai+VLEN*32); 
      BCL_vldu(Va33, Ai+VLEN*33); 
      BCL_vldu(Va34, Ai+VLEN*34); 
      BCL_vldu(Va35, Ai+VLEN*35); 
      BCL_vldu(Va36, Ai+VLEN*36); 
      BCL_vldu(Va37, Ai+VLEN*37); 

      for (INDEXTYPE j = pntrb[i]; j < pntre[i]; j++)
      {
         VTYPE Vb0, Vb1, Vb2, Vb3, Vb4, Vb5, Vb6, Vb7, Vb8, Vb9, Vb10, Vb11,
               Vb12, Vb13, Vb14, Vb15, Vb16, Vb17, Vb18, Vb19, Vb20, Vb21, Vb22,
               Vb23, Vb24, Vb25, Vb26, Vb27, Vb28, Vb29, Vb30, Vb31, Vb32, Vb33,
               Vb34, Vb35, Vb36, Vb37;
         VTYPE Vd0, Vd1; 
         float d1;
         VTYPE Vatt0, Vatt1, Vatt2, Vatt3, Vatt4, Vatt5, Vatt6, Vatt7, Vatt8,
               Vatt9, Vatt10, Vatt11, Vatt12, Vatt13, Vatt14, Vatt15, Vatt16,
               Vatt17, Vatt18, Vatt19, Vatt20, Vatt21, Vatt22, Vatt23, Vatt24,
               Vatt25, Vatt26, Vatt27, Vatt28, Vatt29, Vatt30, Vatt31, Vatt32,
               Vatt33, Vatt34, Vatt35, Vatt36, Vatt37;
         float attrc = 0;
         INDEXTYPE colidj = indx[j];
         INDEXTYPE jindex = colidj*304;
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
         BCL_vldu(Vb16, Bj+VLEN*16); 
         BCL_vldu(Vb17, Bj+VLEN*17); 
         BCL_vldu(Vb18, Bj+VLEN*18); 
         BCL_vldu(Vb19, Bj+VLEN*19); 
         BCL_vldu(Vb20, Bj+VLEN*20); 
         BCL_vldu(Vb21, Bj+VLEN*21); 
         BCL_vldu(Vb22, Bj+VLEN*22); 
         BCL_vldu(Vb23, Bj+VLEN*23); 
         BCL_vldu(Vb24, Bj+VLEN*24); 
         BCL_vldu(Vb25, Bj+VLEN*25); 
         BCL_vldu(Vb26, Bj+VLEN*26); 
         BCL_vldu(Vb27, Bj+VLEN*27); 
         BCL_vldu(Vb28, Bj+VLEN*28); 
         BCL_vldu(Vb29, Bj+VLEN*29); 
         BCL_vldu(Vb30, Bj+VLEN*30); 
         BCL_vldu(Vb31, Bj+VLEN*31); 
         BCL_vldu(Vb32, Bj+VLEN*32); 
         BCL_vldu(Vb33, Bj+VLEN*33); 
         BCL_vldu(Vb34, Bj+VLEN*34); 
         BCL_vldu(Vb35, Bj+VLEN*35); 
         BCL_vldu(Vb36, Bj+VLEN*36); 
         BCL_vldu(Vb37, Bj+VLEN*37); 
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
         BCL_vzero(Vatt16);
         BCL_vzero(Vatt17);
         BCL_vzero(Vatt18);
         BCL_vzero(Vatt19);
         BCL_vzero(Vatt20);
         BCL_vzero(Vatt21);
         BCL_vzero(Vatt22);
         BCL_vzero(Vatt23);
         BCL_vzero(Vatt24);
         BCL_vzero(Vatt25);
         BCL_vzero(Vatt26);
         BCL_vzero(Vatt27);
         BCL_vzero(Vatt28);
         BCL_vzero(Vatt29);
         BCL_vzero(Vatt30);
         BCL_vzero(Vatt31);
         BCL_vzero(Vatt32);
         BCL_vzero(Vatt33);
         BCL_vzero(Vatt34);
         BCL_vzero(Vatt35);
         BCL_vzero(Vatt36);
         BCL_vzero(Vatt37);

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
         BCL_vmac(Vatt16, Va16, Vb16);
         BCL_vmac(Vatt17, Va17, Vb17);
         BCL_vmac(Vatt18, Va18, Vb18);
         BCL_vmac(Vatt19, Va19, Vb19);
         BCL_vmac(Vatt20, Va20, Vb20);
         BCL_vmac(Vatt21, Va21, Vb21);
         BCL_vmac(Vatt22, Va22, Vb22);
         BCL_vmac(Vatt23, Va23, Vb23);
         BCL_vmac(Vatt24, Va24, Vb24);
         BCL_vmac(Vatt25, Va25, Vb25);
         BCL_vmac(Vatt26, Va26, Vb26);
         BCL_vmac(Vatt27, Va27, Vb27);
         BCL_vmac(Vatt28, Va28, Vb28);
         BCL_vmac(Vatt29, Va29, Vb29);
         BCL_vmac(Vatt30, Va30, Vb30);
         BCL_vmac(Vatt31, Va31, Vb31);
         BCL_vmac(Vatt32, Va32, Vb32);
         BCL_vmac(Vatt33, Va33, Vb33);
         BCL_vmac(Vatt34, Va34, Vb34);
         BCL_vmac(Vatt35, Va35, Vb35);
         BCL_vmac(Vatt36, Va36, Vb36);
         BCL_vmac(Vatt37, Va37, Vb37);
         // binary tree reduction 
         BCL_vadd(Vatt0, Vatt0, Vatt1);
         BCL_vadd(Vatt2, Vatt2, Vatt3);
         BCL_vadd(Vatt4, Vatt4, Vatt5);
         BCL_vadd(Vatt6, Vatt6, Vatt7);
         BCL_vadd(Vatt8, Vatt8, Vatt9);
         BCL_vadd(Vatt10, Vatt10, Vatt11);
         BCL_vadd(Vatt12, Vatt12, Vatt13);
         BCL_vadd(Vatt14, Vatt14, Vatt15);
         BCL_vadd(Vatt16, Vatt16, Vatt17);
         BCL_vadd(Vatt18, Vatt18, Vatt19);
         BCL_vadd(Vatt20, Vatt20, Vatt21);
         BCL_vadd(Vatt22, Vatt22, Vatt23);
         BCL_vadd(Vatt24, Vatt24, Vatt25);
         BCL_vadd(Vatt26, Vatt26, Vatt27);
         BCL_vadd(Vatt28, Vatt28, Vatt29);
         BCL_vadd(Vatt30, Vatt30, Vatt31);
         BCL_vadd(Vatt32, Vatt32, Vatt33);
         BCL_vadd(Vatt34, Vatt34, Vatt35);
         BCL_vadd(Vatt36, Vatt36, Vatt37);
         BCL_vadd(Vatt0, Vatt0, Vatt2);
         BCL_vadd(Vatt4, Vatt4, Vatt6);
         BCL_vadd(Vatt8, Vatt8, Vatt10);
         BCL_vadd(Vatt12, Vatt12, Vatt14);
         BCL_vadd(Vatt16, Vatt16, Vatt18);
         BCL_vadd(Vatt20, Vatt20, Vatt22);
         BCL_vadd(Vatt24, Vatt24, Vatt26);
         BCL_vadd(Vatt28, Vatt28, Vatt30);
         BCL_vadd(Vatt32, Vatt32, Vatt34);
         BCL_vadd(Vatt0, Vatt0, Vatt4);
         BCL_vadd(Vatt8, Vatt8, Vatt12);
         BCL_vadd(Vatt16, Vatt16, Vatt20);
         BCL_vadd(Vatt24, Vatt24, Vatt28);
         BCL_vadd(Vatt32, Vatt32, Vatt36);
         BCL_vadd(Vatt0, Vatt0, Vatt8);
         BCL_vadd(Vatt16, Vatt16, Vatt24);
         BCL_vadd(Vatt0, Vatt0, Vatt16);
         BCL_vadd(Vatt0, Vatt0, Vatt32);

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
         BCL_vmac(Vc16, Vd1, Vb16);
         BCL_vmac(Vc17, Vd1, Vb17);
         BCL_vmac(Vc18, Vd1, Vb18);
         BCL_vmac(Vc19, Vd1, Vb19);
         BCL_vmac(Vc20, Vd1, Vb20);
         BCL_vmac(Vc21, Vd1, Vb21);
         BCL_vmac(Vc22, Vd1, Vb22);
         BCL_vmac(Vc23, Vd1, Vb23);
         BCL_vmac(Vc24, Vd1, Vb24);
         BCL_vmac(Vc25, Vd1, Vb25);
         BCL_vmac(Vc26, Vd1, Vb26);
         BCL_vmac(Vc27, Vd1, Vb27);
         BCL_vmac(Vc28, Vd1, Vb28);
         BCL_vmac(Vc29, Vd1, Vb29);
         BCL_vmac(Vc30, Vd1, Vb30);
         BCL_vmac(Vc31, Vd1, Vb31);
         BCL_vmac(Vc32, Vd1, Vb32);
         BCL_vmac(Vc33, Vd1, Vb33);
         BCL_vmac(Vc34, Vd1, Vb34);
         BCL_vmac(Vc35, Vd1, Vb35);
         BCL_vmac(Vc36, Vd1, Vb36);
         BCL_vmac(Vc37, Vd1, Vb37);
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
      BCL_vstu(Ci + VLEN*16, Vc16); 
      BCL_vstu(Ci + VLEN*17, Vc17); 
      BCL_vstu(Ci + VLEN*18, Vc18); 
      BCL_vstu(Ci + VLEN*19, Vc19); 
      BCL_vstu(Ci + VLEN*20, Vc20); 
      BCL_vstu(Ci + VLEN*21, Vc21); 
      BCL_vstu(Ci + VLEN*22, Vc22); 
      BCL_vstu(Ci + VLEN*23, Vc23); 
      BCL_vstu(Ci + VLEN*24, Vc24); 
      BCL_vstu(Ci + VLEN*25, Vc25); 
      BCL_vstu(Ci + VLEN*26, Vc26); 
      BCL_vstu(Ci + VLEN*27, Vc27); 
      BCL_vstu(Ci + VLEN*28, Vc28); 
      BCL_vstu(Ci + VLEN*29, Vc29); 
      BCL_vstu(Ci + VLEN*30, Vc30); 
      BCL_vstu(Ci + VLEN*31, Vc31); 
      BCL_vstu(Ci + VLEN*32, Vc32); 
      BCL_vstu(Ci + VLEN*33, Vc33); 
      BCL_vstu(Ci + VLEN*34, Vc34); 
      BCL_vstu(Ci + VLEN*35, Vc35); 
      BCL_vstu(Ci + VLEN*36, Vc36); 
      BCL_vstu(Ci + VLEN*37, Vc37); 
   }
#if defined(PTTIME) && defined(LDB)
   }
#endif
#ifdef SOP_INHOUSE
   free(sm_table);
#endif
   return 0;
}
#elif DIM == 608
#ifdef BETA0 
__attribute__((noinline)) Value sgfusedMM_K608_sigmoid_b0_csr
#else /* BETA1 version */
__attribute__((noinline)) Value sgfusedMM_K608_sigmoid_b1_csr
#endif
(
   const INDEXTYPE m,      // rows of dense A matrix 
   const INDEXTYPE k,      // cols of A or dimension. not used since K compile time   
   const float alpha,     // const to scale, not use yet  
   const float *val,       // value of  the sparse matrix 
   const INDEXTYPE *indx,  // colids -> column indices of sparse matrix 
   const INDEXTYPE *pntrb, // starting index for rowptr of csr of sparse matrix
   const INDEXTYPE *pntre, // ending index for rowptr of csr of sparse matrix 
   const float *a,        // Dense A matrix
   const INDEXTYPE lda,    // leading dimension of a (col size since row-major)  
   const float *b,        // Dense B matrix
   const INDEXTYPE ldb,    // leading dimension of b (col size since row-major)  
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
                     Vc11, Va12, Vc12, Va13, Vc13, Va14, Vc14, Va15, Vc15, Va16,
                     Vc16, Va17, Vc17, Va18, Vc18, Va19, Vc19, Va20, Vc20, Va21,
                     Vc21, Va22, Vc22, Va23, Vc23, Va24, Vc24, Va25, Vc25, Va26,
                     Vc26, Va27, Vc27, Va28, Vc28, Va29, Vc29, Va30, Vc30, Va31,
                     Vc31, Va32, Vc32, Va33, Vc33, Va34, Vc34, Va35, Vc35, Va36,
                     Vc36, Va37, Vc37, Va38, Vc38, Va39, Vc39, Va40, Vc40, Va41,
                     Vc41, Va42, Vc42, Va43, Vc43, Va44, Vc44, Va45, Vc45, Va46,
                     Vc46, Va47, Vc47, Va48, Vc48, Va49, Vc49, Va50, Vc50, Va51,
                     Vc51, Va52, Vc52, Va53, Vc53, Va54, Vc54, Va55, Vc55, Va56,
                     Vc56, Va57, Vc57, Va58, Vc58, Va59, Vc59, Va60, Vc60, Va61,
                     Vc61, Va62, Vc62, Va63, Vc63, Va64, Vc64, Va65, Vc65, Va66,
                     Vc66, Va67, Vc67, Va68, Vc68, Va69, Vc69, Va70, Vc70, Va71,
                     Vc71, Va72, Vc72, Va73, Vc73, Va74, Vc74, Va75, Vc75;
      INDEXTYPE iindex = i * k; 
      const float *Ai = a + iindex; 
      float *Ci = c + iindex; 
      VTYPE VMAXBOUND, VMINBOUND; 
#ifdef SOP_INHOUSE
      BCL_vset1(VMAXBOUND, sm_bound); 
      BCL_vset1(VMINBOUND, -sm_bound); 
#endif
     INDEXTYPE offset_begin = pntrb[i];
     INDEXTYPE offset_end = pntre[i];
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
      BCL_vzero(Vc16); 
      BCL_vzero(Vc17); 
      BCL_vzero(Vc18); 
      BCL_vzero(Vc19); 
      BCL_vzero(Vc20); 
      BCL_vzero(Vc21); 
      BCL_vzero(Vc22); 
      BCL_vzero(Vc23); 
      BCL_vzero(Vc24); 
      BCL_vzero(Vc25); 
      BCL_vzero(Vc26); 
      BCL_vzero(Vc27); 
      BCL_vzero(Vc28); 
      BCL_vzero(Vc29); 
      BCL_vzero(Vc30); 
      BCL_vzero(Vc31); 
      BCL_vzero(Vc32); 
      BCL_vzero(Vc33); 
      BCL_vzero(Vc34); 
      BCL_vzero(Vc35); 
      BCL_vzero(Vc36); 
      BCL_vzero(Vc37); 
      BCL_vzero(Vc38); 
      BCL_vzero(Vc39); 
      BCL_vzero(Vc40); 
      BCL_vzero(Vc41); 
      BCL_vzero(Vc42); 
      BCL_vzero(Vc43); 
      BCL_vzero(Vc44); 
      BCL_vzero(Vc45); 
      BCL_vzero(Vc46); 
      BCL_vzero(Vc47); 
      BCL_vzero(Vc48); 
      BCL_vzero(Vc49); 
      BCL_vzero(Vc50); 
      BCL_vzero(Vc51); 
      BCL_vzero(Vc52); 
      BCL_vzero(Vc53); 
      BCL_vzero(Vc54); 
      BCL_vzero(Vc55); 
      BCL_vzero(Vc56); 
      BCL_vzero(Vc57); 
      BCL_vzero(Vc58); 
      BCL_vzero(Vc59); 
      BCL_vzero(Vc60); 
      BCL_vzero(Vc61); 
      BCL_vzero(Vc62); 
      BCL_vzero(Vc63); 
      BCL_vzero(Vc64); 
      BCL_vzero(Vc65); 
      BCL_vzero(Vc66); 
      BCL_vzero(Vc67); 
      BCL_vzero(Vc68); 
      BCL_vzero(Vc69); 
      BCL_vzero(Vc70); 
      BCL_vzero(Vc71); 
      BCL_vzero(Vc72); 
      BCL_vzero(Vc73); 
      BCL_vzero(Vc74); 
      BCL_vzero(Vc75); 

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
      BCL_vldu(Vc16, Ci+VLEN*16); 
      BCL_vldu(Vc17, Ci+VLEN*17); 
      BCL_vldu(Vc18, Ci+VLEN*18); 
      BCL_vldu(Vc19, Ci+VLEN*19); 
      BCL_vldu(Vc20, Ci+VLEN*20); 
      BCL_vldu(Vc21, Ci+VLEN*21); 
      BCL_vldu(Vc22, Ci+VLEN*22); 
      BCL_vldu(Vc23, Ci+VLEN*23); 
      BCL_vldu(Vc24, Ci+VLEN*24); 
      BCL_vldu(Vc25, Ci+VLEN*25); 
      BCL_vldu(Vc26, Ci+VLEN*26); 
      BCL_vldu(Vc27, Ci+VLEN*27); 
      BCL_vldu(Vc28, Ci+VLEN*28); 
      BCL_vldu(Vc29, Ci+VLEN*29); 
      BCL_vldu(Vc30, Ci+VLEN*30); 
      BCL_vldu(Vc31, Ci+VLEN*31); 
      BCL_vldu(Vc32, Ci+VLEN*32); 
      BCL_vldu(Vc33, Ci+VLEN*33); 
      BCL_vldu(Vc34, Ci+VLEN*34); 
      BCL_vldu(Vc35, Ci+VLEN*35); 
      BCL_vldu(Vc36, Ci+VLEN*36); 
      BCL_vldu(Vc37, Ci+VLEN*37); 
      BCL_vldu(Vc38, Ci+VLEN*38); 
      BCL_vldu(Vc39, Ci+VLEN*39); 
      BCL_vldu(Vc40, Ci+VLEN*40); 
      BCL_vldu(Vc41, Ci+VLEN*41); 
      BCL_vldu(Vc42, Ci+VLEN*42); 
      BCL_vldu(Vc43, Ci+VLEN*43); 
      BCL_vldu(Vc44, Ci+VLEN*44); 
      BCL_vldu(Vc45, Ci+VLEN*45); 
      BCL_vldu(Vc46, Ci+VLEN*46); 
      BCL_vldu(Vc47, Ci+VLEN*47); 
      BCL_vldu(Vc48, Ci+VLEN*48); 
      BCL_vldu(Vc49, Ci+VLEN*49); 
      BCL_vldu(Vc50, Ci+VLEN*50); 
      BCL_vldu(Vc51, Ci+VLEN*51); 
      BCL_vldu(Vc52, Ci+VLEN*52); 
      BCL_vldu(Vc53, Ci+VLEN*53); 
      BCL_vldu(Vc54, Ci+VLEN*54); 
      BCL_vldu(Vc55, Ci+VLEN*55); 
      BCL_vldu(Vc56, Ci+VLEN*56); 
      BCL_vldu(Vc57, Ci+VLEN*57); 
      BCL_vldu(Vc58, Ci+VLEN*58); 
      BCL_vldu(Vc59, Ci+VLEN*59); 
      BCL_vldu(Vc60, Ci+VLEN*60); 
      BCL_vldu(Vc61, Ci+VLEN*61); 
      BCL_vldu(Vc62, Ci+VLEN*62); 
      BCL_vldu(Vc63, Ci+VLEN*63); 
      BCL_vldu(Vc64, Ci+VLEN*64); 
      BCL_vldu(Vc65, Ci+VLEN*65); 
      BCL_vldu(Vc66, Ci+VLEN*66); 
      BCL_vldu(Vc67, Ci+VLEN*67); 
      BCL_vldu(Vc68, Ci+VLEN*68); 
      BCL_vldu(Vc69, Ci+VLEN*69); 
      BCL_vldu(Vc70, Ci+VLEN*70); 
      BCL_vldu(Vc71, Ci+VLEN*71); 
      BCL_vldu(Vc72, Ci+VLEN*72); 
      BCL_vldu(Vc73, Ci+VLEN*73); 
      BCL_vldu(Vc74, Ci+VLEN*74); 
      BCL_vldu(Vc75, Ci+VLEN*75); 
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
      BCL_vldu(Va16, Ai+VLEN*16); 
      BCL_vldu(Va17, Ai+VLEN*17); 
      BCL_vldu(Va18, Ai+VLEN*18); 
      BCL_vldu(Va19, Ai+VLEN*19); 
      BCL_vldu(Va20, Ai+VLEN*20); 
      BCL_vldu(Va21, Ai+VLEN*21); 
      BCL_vldu(Va22, Ai+VLEN*22); 
      BCL_vldu(Va23, Ai+VLEN*23); 
      BCL_vldu(Va24, Ai+VLEN*24); 
      BCL_vldu(Va25, Ai+VLEN*25); 
      BCL_vldu(Va26, Ai+VLEN*26); 
      BCL_vldu(Va27, Ai+VLEN*27); 
      BCL_vldu(Va28, Ai+VLEN*28); 
      BCL_vldu(Va29, Ai+VLEN*29); 
      BCL_vldu(Va30, Ai+VLEN*30); 
      BCL_vldu(Va31, Ai+VLEN*31); 
      BCL_vldu(Va32, Ai+VLEN*32); 
      BCL_vldu(Va33, Ai+VLEN*33); 
      BCL_vldu(Va34, Ai+VLEN*34); 
      BCL_vldu(Va35, Ai+VLEN*35); 
      BCL_vldu(Va36, Ai+VLEN*36); 
      BCL_vldu(Va37, Ai+VLEN*37); 
      BCL_vldu(Va38, Ai+VLEN*38); 
      BCL_vldu(Va39, Ai+VLEN*39); 
      BCL_vldu(Va40, Ai+VLEN*40); 
      BCL_vldu(Va41, Ai+VLEN*41); 
      BCL_vldu(Va42, Ai+VLEN*42); 
      BCL_vldu(Va43, Ai+VLEN*43); 
      BCL_vldu(Va44, Ai+VLEN*44); 
      BCL_vldu(Va45, Ai+VLEN*45); 
      BCL_vldu(Va46, Ai+VLEN*46); 
      BCL_vldu(Va47, Ai+VLEN*47); 
      BCL_vldu(Va48, Ai+VLEN*48); 
      BCL_vldu(Va49, Ai+VLEN*49); 
      BCL_vldu(Va50, Ai+VLEN*50); 
      BCL_vldu(Va51, Ai+VLEN*51); 
      BCL_vldu(Va52, Ai+VLEN*52); 
      BCL_vldu(Va53, Ai+VLEN*53); 
      BCL_vldu(Va54, Ai+VLEN*54); 
      BCL_vldu(Va55, Ai+VLEN*55); 
      BCL_vldu(Va56, Ai+VLEN*56); 
      BCL_vldu(Va57, Ai+VLEN*57); 
      BCL_vldu(Va58, Ai+VLEN*58); 
      BCL_vldu(Va59, Ai+VLEN*59); 
      BCL_vldu(Va60, Ai+VLEN*60); 
      BCL_vldu(Va61, Ai+VLEN*61); 
      BCL_vldu(Va62, Ai+VLEN*62); 
      BCL_vldu(Va63, Ai+VLEN*63); 
      BCL_vldu(Va64, Ai+VLEN*64); 
      BCL_vldu(Va65, Ai+VLEN*65); 
      BCL_vldu(Va66, Ai+VLEN*66); 
      BCL_vldu(Va67, Ai+VLEN*67); 
      BCL_vldu(Va68, Ai+VLEN*68); 
      BCL_vldu(Va69, Ai+VLEN*69); 
      BCL_vldu(Va70, Ai+VLEN*70); 
      BCL_vldu(Va71, Ai+VLEN*71); 
      BCL_vldu(Va72, Ai+VLEN*72); 
      BCL_vldu(Va73, Ai+VLEN*73); 
      BCL_vldu(Va74, Ai+VLEN*74); 
      BCL_vldu(Va75, Ai+VLEN*75); 

      for (INDEXTYPE j = pntrb[i]; j < pntre[i]; j++)
      {
         VTYPE Vb0, Vb1, Vb2, Vb3, Vb4, Vb5, Vb6, Vb7, Vb8, Vb9, Vb10, Vb11,
               Vb12, Vb13, Vb14, Vb15, Vb16, Vb17, Vb18, Vb19, Vb20, Vb21, Vb22,
               Vb23, Vb24, Vb25, Vb26, Vb27, Vb28, Vb29, Vb30, Vb31, Vb32, Vb33,
               Vb34, Vb35, Vb36, Vb37, Vb38, Vb39, Vb40, Vb41, Vb42, Vb43, Vb44,
               Vb45, Vb46, Vb47, Vb48, Vb49, Vb50, Vb51, Vb52, Vb53, Vb54, Vb55,
               Vb56, Vb57, Vb58, Vb59, Vb60, Vb61, Vb62, Vb63, Vb64, Vb65, Vb66,
               Vb67, Vb68, Vb69, Vb70, Vb71, Vb72, Vb73, Vb74, Vb75;
         VTYPE Vd0, Vd1; 
         float d1;
         VTYPE Vatt0, Vatt1, Vatt2, Vatt3, Vatt4, Vatt5, Vatt6, Vatt7, Vatt8,
               Vatt9, Vatt10, Vatt11, Vatt12, Vatt13, Vatt14, Vatt15, Vatt16,
               Vatt17, Vatt18, Vatt19, Vatt20, Vatt21, Vatt22, Vatt23, Vatt24,
               Vatt25, Vatt26, Vatt27, Vatt28, Vatt29, Vatt30, Vatt31, Vatt32,
               Vatt33, Vatt34, Vatt35, Vatt36, Vatt37, Vatt38, Vatt39, Vatt40,
               Vatt41, Vatt42, Vatt43, Vatt44, Vatt45, Vatt46, Vatt47, Vatt48,
               Vatt49, Vatt50, Vatt51, Vatt52, Vatt53, Vatt54, Vatt55, Vatt56,
               Vatt57, Vatt58, Vatt59, Vatt60, Vatt61, Vatt62, Vatt63, Vatt64,
               Vatt65, Vatt66, Vatt67, Vatt68, Vatt69, Vatt70, Vatt71, Vatt72,
               Vatt73, Vatt74, Vatt75;
         float attrc = 0;
         INDEXTYPE colidj = indx[j];
         INDEXTYPE jindex = colidj*k;
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
         BCL_vldu(Vb16, Bj+VLEN*16); 
         BCL_vldu(Vb17, Bj+VLEN*17); 
         BCL_vldu(Vb18, Bj+VLEN*18); 
         BCL_vldu(Vb19, Bj+VLEN*19); 
         BCL_vldu(Vb20, Bj+VLEN*20); 
         BCL_vldu(Vb21, Bj+VLEN*21); 
         BCL_vldu(Vb22, Bj+VLEN*22); 
         BCL_vldu(Vb23, Bj+VLEN*23); 
         BCL_vldu(Vb24, Bj+VLEN*24); 
         BCL_vldu(Vb25, Bj+VLEN*25); 
         BCL_vldu(Vb26, Bj+VLEN*26); 
         BCL_vldu(Vb27, Bj+VLEN*27); 
         BCL_vldu(Vb28, Bj+VLEN*28); 
         BCL_vldu(Vb29, Bj+VLEN*29); 
         BCL_vldu(Vb30, Bj+VLEN*30); 
         BCL_vldu(Vb31, Bj+VLEN*31); 
         BCL_vldu(Vb32, Bj+VLEN*32); 
         BCL_vldu(Vb33, Bj+VLEN*33); 
         BCL_vldu(Vb34, Bj+VLEN*34); 
         BCL_vldu(Vb35, Bj+VLEN*35); 
         BCL_vldu(Vb36, Bj+VLEN*36); 
         BCL_vldu(Vb37, Bj+VLEN*37); 
         BCL_vldu(Vb38, Bj+VLEN*38); 
         BCL_vldu(Vb39, Bj+VLEN*39); 
         BCL_vldu(Vb40, Bj+VLEN*40); 
         BCL_vldu(Vb41, Bj+VLEN*41); 
         BCL_vldu(Vb42, Bj+VLEN*42); 
         BCL_vldu(Vb43, Bj+VLEN*43); 
         BCL_vldu(Vb44, Bj+VLEN*44); 
         BCL_vldu(Vb45, Bj+VLEN*45); 
         BCL_vldu(Vb46, Bj+VLEN*46); 
         BCL_vldu(Vb47, Bj+VLEN*47); 
         BCL_vldu(Vb48, Bj+VLEN*48); 
         BCL_vldu(Vb49, Bj+VLEN*49); 
         BCL_vldu(Vb50, Bj+VLEN*50); 
         BCL_vldu(Vb51, Bj+VLEN*51); 
         BCL_vldu(Vb52, Bj+VLEN*52); 
         BCL_vldu(Vb53, Bj+VLEN*53); 
         BCL_vldu(Vb54, Bj+VLEN*54); 
         BCL_vldu(Vb55, Bj+VLEN*55); 
         BCL_vldu(Vb56, Bj+VLEN*56); 
         BCL_vldu(Vb57, Bj+VLEN*57); 
         BCL_vldu(Vb58, Bj+VLEN*58); 
         BCL_vldu(Vb59, Bj+VLEN*59); 
         BCL_vldu(Vb60, Bj+VLEN*60); 
         BCL_vldu(Vb61, Bj+VLEN*61); 
         BCL_vldu(Vb62, Bj+VLEN*62); 
         BCL_vldu(Vb63, Bj+VLEN*63); 
         BCL_vldu(Vb64, Bj+VLEN*64); 
         BCL_vldu(Vb65, Bj+VLEN*65); 
         BCL_vldu(Vb66, Bj+VLEN*66); 
         BCL_vldu(Vb67, Bj+VLEN*67); 
         BCL_vldu(Vb68, Bj+VLEN*68); 
         BCL_vldu(Vb69, Bj+VLEN*69); 
         BCL_vldu(Vb70, Bj+VLEN*70); 
         BCL_vldu(Vb71, Bj+VLEN*71); 
         BCL_vldu(Vb72, Bj+VLEN*72); 
         BCL_vldu(Vb73, Bj+VLEN*73); 
         BCL_vldu(Vb74, Bj+VLEN*74); 
         BCL_vldu(Vb75, Bj+VLEN*75); 
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
         BCL_vzero(Vatt16);
         BCL_vzero(Vatt17);
         BCL_vzero(Vatt18);
         BCL_vzero(Vatt19);
         BCL_vzero(Vatt20);
         BCL_vzero(Vatt21);
         BCL_vzero(Vatt22);
         BCL_vzero(Vatt23);
         BCL_vzero(Vatt24);
         BCL_vzero(Vatt25);
         BCL_vzero(Vatt26);
         BCL_vzero(Vatt27);
         BCL_vzero(Vatt28);
         BCL_vzero(Vatt29);
         BCL_vzero(Vatt30);
         BCL_vzero(Vatt31);
         BCL_vzero(Vatt32);
         BCL_vzero(Vatt33);
         BCL_vzero(Vatt34);
         BCL_vzero(Vatt35);
         BCL_vzero(Vatt36);
         BCL_vzero(Vatt37);
         BCL_vzero(Vatt38);
         BCL_vzero(Vatt39);
         BCL_vzero(Vatt40);
         BCL_vzero(Vatt41);
         BCL_vzero(Vatt42);
         BCL_vzero(Vatt43);
         BCL_vzero(Vatt44);
         BCL_vzero(Vatt45);
         BCL_vzero(Vatt46);
         BCL_vzero(Vatt47);
         BCL_vzero(Vatt48);
         BCL_vzero(Vatt49);
         BCL_vzero(Vatt50);
         BCL_vzero(Vatt51);
         BCL_vzero(Vatt52);
         BCL_vzero(Vatt53);
         BCL_vzero(Vatt54);
         BCL_vzero(Vatt55);
         BCL_vzero(Vatt56);
         BCL_vzero(Vatt57);
         BCL_vzero(Vatt58);
         BCL_vzero(Vatt59);
         BCL_vzero(Vatt60);
         BCL_vzero(Vatt61);
         BCL_vzero(Vatt62);
         BCL_vzero(Vatt63);
         BCL_vzero(Vatt64);
         BCL_vzero(Vatt65);
         BCL_vzero(Vatt66);
         BCL_vzero(Vatt67);
         BCL_vzero(Vatt68);
         BCL_vzero(Vatt69);
         BCL_vzero(Vatt70);
         BCL_vzero(Vatt71);
         BCL_vzero(Vatt72);
         BCL_vzero(Vatt73);
         BCL_vzero(Vatt74);
         BCL_vzero(Vatt75);

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
         BCL_vmac(Vatt16, Va16, Vb16);
         BCL_vmac(Vatt17, Va17, Vb17);
         BCL_vmac(Vatt18, Va18, Vb18);
         BCL_vmac(Vatt19, Va19, Vb19);
         BCL_vmac(Vatt20, Va20, Vb20);
         BCL_vmac(Vatt21, Va21, Vb21);
         BCL_vmac(Vatt22, Va22, Vb22);
         BCL_vmac(Vatt23, Va23, Vb23);
         BCL_vmac(Vatt24, Va24, Vb24);
         BCL_vmac(Vatt25, Va25, Vb25);
         BCL_vmac(Vatt26, Va26, Vb26);
         BCL_vmac(Vatt27, Va27, Vb27);
         BCL_vmac(Vatt28, Va28, Vb28);
         BCL_vmac(Vatt29, Va29, Vb29);
         BCL_vmac(Vatt30, Va30, Vb30);
         BCL_vmac(Vatt31, Va31, Vb31);
         BCL_vmac(Vatt32, Va32, Vb32);
         BCL_vmac(Vatt33, Va33, Vb33);
         BCL_vmac(Vatt34, Va34, Vb34);
         BCL_vmac(Vatt35, Va35, Vb35);
         BCL_vmac(Vatt36, Va36, Vb36);
         BCL_vmac(Vatt37, Va37, Vb37);
         BCL_vmac(Vatt38, Va38, Vb38);
         BCL_vmac(Vatt39, Va39, Vb39);
         BCL_vmac(Vatt40, Va40, Vb40);
         BCL_vmac(Vatt41, Va41, Vb41);
         BCL_vmac(Vatt42, Va42, Vb42);
         BCL_vmac(Vatt43, Va43, Vb43);
         BCL_vmac(Vatt44, Va44, Vb44);
         BCL_vmac(Vatt45, Va45, Vb45);
         BCL_vmac(Vatt46, Va46, Vb46);
         BCL_vmac(Vatt47, Va47, Vb47);
         BCL_vmac(Vatt48, Va48, Vb48);
         BCL_vmac(Vatt49, Va49, Vb49);
         BCL_vmac(Vatt50, Va50, Vb50);
         BCL_vmac(Vatt51, Va51, Vb51);
         BCL_vmac(Vatt52, Va52, Vb52);
         BCL_vmac(Vatt53, Va53, Vb53);
         BCL_vmac(Vatt54, Va54, Vb54);
         BCL_vmac(Vatt55, Va55, Vb55);
         BCL_vmac(Vatt56, Va56, Vb56);
         BCL_vmac(Vatt57, Va57, Vb57);
         BCL_vmac(Vatt58, Va58, Vb58);
         BCL_vmac(Vatt59, Va59, Vb59);
         BCL_vmac(Vatt60, Va60, Vb60);
         BCL_vmac(Vatt61, Va61, Vb61);
         BCL_vmac(Vatt62, Va62, Vb62);
         BCL_vmac(Vatt63, Va63, Vb63);
         BCL_vmac(Vatt64, Va64, Vb64);
         BCL_vmac(Vatt65, Va65, Vb65);
         BCL_vmac(Vatt66, Va66, Vb66);
         BCL_vmac(Vatt67, Va67, Vb67);
         BCL_vmac(Vatt68, Va68, Vb68);
         BCL_vmac(Vatt69, Va69, Vb69);
         BCL_vmac(Vatt70, Va70, Vb70);
         BCL_vmac(Vatt71, Va71, Vb71);
         BCL_vmac(Vatt72, Va72, Vb72);
         BCL_vmac(Vatt73, Va73, Vb73);
         BCL_vmac(Vatt74, Va74, Vb74);
         BCL_vmac(Vatt75, Va75, Vb75);
         // binary tree reduction 
         BCL_vadd(Vatt0, Vatt0, Vatt1);
         BCL_vadd(Vatt2, Vatt2, Vatt3);
         BCL_vadd(Vatt4, Vatt4, Vatt5);
         BCL_vadd(Vatt6, Vatt6, Vatt7);
         BCL_vadd(Vatt8, Vatt8, Vatt9);
         BCL_vadd(Vatt10, Vatt10, Vatt11);
         BCL_vadd(Vatt12, Vatt12, Vatt13);
         BCL_vadd(Vatt14, Vatt14, Vatt15);
         BCL_vadd(Vatt16, Vatt16, Vatt17);
         BCL_vadd(Vatt18, Vatt18, Vatt19);
         BCL_vadd(Vatt20, Vatt20, Vatt21);
         BCL_vadd(Vatt22, Vatt22, Vatt23);
         BCL_vadd(Vatt24, Vatt24, Vatt25);
         BCL_vadd(Vatt26, Vatt26, Vatt27);
         BCL_vadd(Vatt28, Vatt28, Vatt29);
         BCL_vadd(Vatt30, Vatt30, Vatt31);
         BCL_vadd(Vatt32, Vatt32, Vatt33);
         BCL_vadd(Vatt34, Vatt34, Vatt35);
         BCL_vadd(Vatt36, Vatt36, Vatt37);
         BCL_vadd(Vatt38, Vatt38, Vatt39);
         BCL_vadd(Vatt40, Vatt40, Vatt41);
         BCL_vadd(Vatt42, Vatt42, Vatt43);
         BCL_vadd(Vatt44, Vatt44, Vatt45);
         BCL_vadd(Vatt46, Vatt46, Vatt47);
         BCL_vadd(Vatt48, Vatt48, Vatt49);
         BCL_vadd(Vatt50, Vatt50, Vatt51);
         BCL_vadd(Vatt52, Vatt52, Vatt53);
         BCL_vadd(Vatt54, Vatt54, Vatt55);
         BCL_vadd(Vatt56, Vatt56, Vatt57);
         BCL_vadd(Vatt58, Vatt58, Vatt59);
         BCL_vadd(Vatt60, Vatt60, Vatt61);
         BCL_vadd(Vatt62, Vatt62, Vatt63);
         BCL_vadd(Vatt64, Vatt64, Vatt65);
         BCL_vadd(Vatt66, Vatt66, Vatt67);
         BCL_vadd(Vatt68, Vatt68, Vatt69);
         BCL_vadd(Vatt70, Vatt70, Vatt71);
         BCL_vadd(Vatt72, Vatt72, Vatt73);
         BCL_vadd(Vatt74, Vatt74, Vatt75);
         BCL_vadd(Vatt0, Vatt0, Vatt2);
         BCL_vadd(Vatt4, Vatt4, Vatt6);
         BCL_vadd(Vatt8, Vatt8, Vatt10);
         BCL_vadd(Vatt12, Vatt12, Vatt14);
         BCL_vadd(Vatt16, Vatt16, Vatt18);
         BCL_vadd(Vatt20, Vatt20, Vatt22);
         BCL_vadd(Vatt24, Vatt24, Vatt26);
         BCL_vadd(Vatt28, Vatt28, Vatt30);
         BCL_vadd(Vatt32, Vatt32, Vatt34);
         BCL_vadd(Vatt36, Vatt36, Vatt38);
         BCL_vadd(Vatt40, Vatt40, Vatt42);
         BCL_vadd(Vatt44, Vatt44, Vatt46);
         BCL_vadd(Vatt48, Vatt48, Vatt50);
         BCL_vadd(Vatt52, Vatt52, Vatt54);
         BCL_vadd(Vatt56, Vatt56, Vatt58);
         BCL_vadd(Vatt60, Vatt60, Vatt62);
         BCL_vadd(Vatt64, Vatt64, Vatt66);
         BCL_vadd(Vatt68, Vatt68, Vatt70);
         BCL_vadd(Vatt72, Vatt72, Vatt74);
         BCL_vadd(Vatt0, Vatt0, Vatt4);
         BCL_vadd(Vatt8, Vatt8, Vatt12);
         BCL_vadd(Vatt16, Vatt16, Vatt20);
         BCL_vadd(Vatt24, Vatt24, Vatt28);
         BCL_vadd(Vatt32, Vatt32, Vatt36);
         BCL_vadd(Vatt40, Vatt40, Vatt44);
         BCL_vadd(Vatt48, Vatt48, Vatt52);
         BCL_vadd(Vatt56, Vatt56, Vatt60);
         BCL_vadd(Vatt64, Vatt64, Vatt68);
         BCL_vadd(Vatt0, Vatt0, Vatt8);
         BCL_vadd(Vatt16, Vatt16, Vatt24);
         BCL_vadd(Vatt32, Vatt32, Vatt40);
         BCL_vadd(Vatt48, Vatt48, Vatt56);
         BCL_vadd(Vatt64, Vatt64, Vatt72);
         BCL_vadd(Vatt0, Vatt0, Vatt16);
         BCL_vadd(Vatt32, Vatt32, Vatt48);
         BCL_vadd(Vatt0, Vatt0, Vatt32);
         BCL_vadd(Vatt0, Vatt0, Vatt64);

         BCL_vrsum_syo(attrc, Vatt0);
         // rolled loop for remaining computation
         for (INDEXTYPE kk=608; kk < k; kk++)
            attrc += Ai[kk] * Bj[kk];   
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
         BCL_vmac(Vc16, Vd1, Vb16);
         BCL_vmac(Vc17, Vd1, Vb17);
         BCL_vmac(Vc18, Vd1, Vb18);
         BCL_vmac(Vc19, Vd1, Vb19);
         BCL_vmac(Vc20, Vd1, Vb20);
         BCL_vmac(Vc21, Vd1, Vb21);
         BCL_vmac(Vc22, Vd1, Vb22);
         BCL_vmac(Vc23, Vd1, Vb23);
         BCL_vmac(Vc24, Vd1, Vb24);
         BCL_vmac(Vc25, Vd1, Vb25);
         BCL_vmac(Vc26, Vd1, Vb26);
         BCL_vmac(Vc27, Vd1, Vb27);
         BCL_vmac(Vc28, Vd1, Vb28);
         BCL_vmac(Vc29, Vd1, Vb29);
         BCL_vmac(Vc30, Vd1, Vb30);
         BCL_vmac(Vc31, Vd1, Vb31);
         BCL_vmac(Vc32, Vd1, Vb32);
         BCL_vmac(Vc33, Vd1, Vb33);
         BCL_vmac(Vc34, Vd1, Vb34);
         BCL_vmac(Vc35, Vd1, Vb35);
         BCL_vmac(Vc36, Vd1, Vb36);
         BCL_vmac(Vc37, Vd1, Vb37);
         BCL_vmac(Vc38, Vd1, Vb38);
         BCL_vmac(Vc39, Vd1, Vb39);
         BCL_vmac(Vc40, Vd1, Vb40);
         BCL_vmac(Vc41, Vd1, Vb41);
         BCL_vmac(Vc42, Vd1, Vb42);
         BCL_vmac(Vc43, Vd1, Vb43);
         BCL_vmac(Vc44, Vd1, Vb44);
         BCL_vmac(Vc45, Vd1, Vb45);
         BCL_vmac(Vc46, Vd1, Vb46);
         BCL_vmac(Vc47, Vd1, Vb47);
         BCL_vmac(Vc48, Vd1, Vb48);
         BCL_vmac(Vc49, Vd1, Vb49);
         BCL_vmac(Vc50, Vd1, Vb50);
         BCL_vmac(Vc51, Vd1, Vb51);
         BCL_vmac(Vc52, Vd1, Vb52);
         BCL_vmac(Vc53, Vd1, Vb53);
         BCL_vmac(Vc54, Vd1, Vb54);
         BCL_vmac(Vc55, Vd1, Vb55);
         BCL_vmac(Vc56, Vd1, Vb56);
         BCL_vmac(Vc57, Vd1, Vb57);
         BCL_vmac(Vc58, Vd1, Vb58);
         BCL_vmac(Vc59, Vd1, Vb59);
         BCL_vmac(Vc60, Vd1, Vb60);
         BCL_vmac(Vc61, Vd1, Vb61);
         BCL_vmac(Vc62, Vd1, Vb62);
         BCL_vmac(Vc63, Vd1, Vb63);
         BCL_vmac(Vc64, Vd1, Vb64);
         BCL_vmac(Vc65, Vd1, Vb65);
         BCL_vmac(Vc66, Vd1, Vb66);
         BCL_vmac(Vc67, Vd1, Vb67);
         BCL_vmac(Vc68, Vd1, Vb68);
         BCL_vmac(Vc69, Vd1, Vb69);
         BCL_vmac(Vc70, Vd1, Vb70);
         BCL_vmac(Vc71, Vd1, Vb71);
         BCL_vmac(Vc72, Vd1, Vb72);
         BCL_vmac(Vc73, Vd1, Vb73);
         BCL_vmac(Vc74, Vd1, Vb74);
         BCL_vmac(Vc75, Vd1, Vb75);
         // rolled loop for remaining C write 
         for (INDEXTYPE kk=608; kk < k; kk++)
            Ci[kk] += d1 * Bj[kk];   
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
      BCL_vstu(Ci + VLEN*16, Vc16); 
      BCL_vstu(Ci + VLEN*17, Vc17); 
      BCL_vstu(Ci + VLEN*18, Vc18); 
      BCL_vstu(Ci + VLEN*19, Vc19); 
      BCL_vstu(Ci + VLEN*20, Vc20); 
      BCL_vstu(Ci + VLEN*21, Vc21); 
      BCL_vstu(Ci + VLEN*22, Vc22); 
      BCL_vstu(Ci + VLEN*23, Vc23); 
      BCL_vstu(Ci + VLEN*24, Vc24); 
      BCL_vstu(Ci + VLEN*25, Vc25); 
      BCL_vstu(Ci + VLEN*26, Vc26); 
      BCL_vstu(Ci + VLEN*27, Vc27); 
      BCL_vstu(Ci + VLEN*28, Vc28); 
      BCL_vstu(Ci + VLEN*29, Vc29); 
      BCL_vstu(Ci + VLEN*30, Vc30); 
      BCL_vstu(Ci + VLEN*31, Vc31); 
      BCL_vstu(Ci + VLEN*32, Vc32); 
      BCL_vstu(Ci + VLEN*33, Vc33); 
      BCL_vstu(Ci + VLEN*34, Vc34); 
      BCL_vstu(Ci + VLEN*35, Vc35); 
      BCL_vstu(Ci + VLEN*36, Vc36); 
      BCL_vstu(Ci + VLEN*37, Vc37); 
      BCL_vstu(Ci + VLEN*38, Vc38); 
      BCL_vstu(Ci + VLEN*39, Vc39); 
      BCL_vstu(Ci + VLEN*40, Vc40); 
      BCL_vstu(Ci + VLEN*41, Vc41); 
      BCL_vstu(Ci + VLEN*42, Vc42); 
      BCL_vstu(Ci + VLEN*43, Vc43); 
      BCL_vstu(Ci + VLEN*44, Vc44); 
      BCL_vstu(Ci + VLEN*45, Vc45); 
      BCL_vstu(Ci + VLEN*46, Vc46); 
      BCL_vstu(Ci + VLEN*47, Vc47); 
      BCL_vstu(Ci + VLEN*48, Vc48); 
      BCL_vstu(Ci + VLEN*49, Vc49); 
      BCL_vstu(Ci + VLEN*50, Vc50); 
      BCL_vstu(Ci + VLEN*51, Vc51); 
      BCL_vstu(Ci + VLEN*52, Vc52); 
      BCL_vstu(Ci + VLEN*53, Vc53); 
      BCL_vstu(Ci + VLEN*54, Vc54); 
      BCL_vstu(Ci + VLEN*55, Vc55); 
      BCL_vstu(Ci + VLEN*56, Vc56); 
      BCL_vstu(Ci + VLEN*57, Vc57); 
      BCL_vstu(Ci + VLEN*58, Vc58); 
      BCL_vstu(Ci + VLEN*59, Vc59); 
      BCL_vstu(Ci + VLEN*60, Vc60); 
      BCL_vstu(Ci + VLEN*61, Vc61); 
      BCL_vstu(Ci + VLEN*62, Vc62); 
      BCL_vstu(Ci + VLEN*63, Vc63); 
      BCL_vstu(Ci + VLEN*64, Vc64); 
      BCL_vstu(Ci + VLEN*65, Vc65); 
      BCL_vstu(Ci + VLEN*66, Vc66); 
      BCL_vstu(Ci + VLEN*67, Vc67); 
      BCL_vstu(Ci + VLEN*68, Vc68); 
      BCL_vstu(Ci + VLEN*69, Vc69); 
      BCL_vstu(Ci + VLEN*70, Vc70); 
      BCL_vstu(Ci + VLEN*71, Vc71); 
      BCL_vstu(Ci + VLEN*72, Vc72); 
      BCL_vstu(Ci + VLEN*73, Vc73); 
      BCL_vstu(Ci + VLEN*74, Vc74); 
      BCL_vstu(Ci + VLEN*75, Vc75); 
   }
#if defined(PTTIME) && defined(LDB)
   }
#endif
#ifdef SOP_INHOUSE
   free(sm_table);
#endif
   return 0;
}
#endif

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

  char *ptr = NULL;
  char dataset_path[256] = "";
  char dataset_name[256] = "";
  char filename[100];
  char input_path[256] = "";

  uint64_t nonzero;
  uint64_t sz;
  uint64_t num_node;
  uint64_t total_num_node;
  uint64_t num_dim;

  strcpy(input_path,argv[2]);
  ptr = strrchr(input_path, '/');
  if(ptr==NULL){
  	strcpy(dataset_name,input_path);
  }
  else{
	strcpy(dataset_name,ptr+1);
  }
  //printf("dataset_name = %s\n",dataset_name); 
  strcpy(dataset_path, input_path);
  //printf("dataset_path = %s\n", dataset_path);
  strcat(dataset_path, "/");
  printf("dataset_path = %s\n", dataset_path);
  strcat(dataset_path, dataset_name);

  // ===============================================================================//
  // indx from file
  strcpy(filename, dataset_path);
  strcat(filename, "_rows.dat");
  printf("file name = %s\n", filename);

  FILE* fp_mtx = fopen(filename, "rb");
  if (fp_mtx != NULL) {
    fseek(fp_mtx, 0L, SEEK_END);
    sz = ftell(fp_mtx);
    fseek(fp_mtx, 0L, SEEK_SET);
	fread((void*)&total_num_node, sizeof(INDEXTYPE), 1, fp_mtx);	
	fread((void*)&nonzero, sizeof(INDEXTYPE), 1, fp_mtx);	
  }
  else {
    printf("Cannot find %s\n", filename);
    return 0;
  }

  //printf("nonzero = %d, total_num_node = %d\n",nonzero, total_num_node);
  INDEXTYPE* indx = (INDEXTYPE*) aligned_alloc(CACHE_LINE_SIZE,  nonzero * sizeof(INDEXTYPE));

  if (sz == (nonzero+2) * sizeof(INDEXTYPE)) {
    fread((void*)indx, sizeof(INDEXTYPE), nonzero, fp_mtx);
  }
  else {
	  printf("size of file(%s) is wrong\n", filename);
	  return 0;
  }
  fclose(fp_mtx);

  //printf("indx[0] = %lu\n" ,indx[0]);
  //printf("indx[1] = %lu\n" ,indx[1]);
  //printf("indx[2] = %lu\n" ,indx[2]);
  //printf("indx[3] = %lu\n" ,indx[3]);
  // ===============================================================================//


  // ===============================================================================//
  // b from file
  strcpy(filename, dataset_path);
  strcat(filename, "_b_mat.dat");
  printf("file name = %s\n", filename);
  FILE* fp_mtx2 = fopen(filename, "rb");
  if (fp_mtx2 != NULL) {
    fseek(fp_mtx2, 0L, SEEK_END);
    sz = ftell(fp_mtx2);	
    fseek(fp_mtx2, 0L, SEEK_SET);
	fread((void*)&num_dim, sizeof(uint64_t), 1, fp_mtx2);	
    //printf("sz = %d, num_dim = %d\n", sz, num_dim);
  }
  else {
    printf("Cannot find %s\n", filename);
    return 0;
  }

  num_node = (sz-sizeof(uint64_t))/(sizeof(VALUETYPE)*num_dim);
  //printf("sz = %d, num_dim = %d, num_node = %d\n", sz, num_dim, num_node);
  VALUETYPE* b = (VALUETYPE*) aligned_alloc(CACHE_LINE_SIZE,  num_node*num_dim * sizeof(VALUETYPE));
  INDEXTYPE ldb = num_dim;
  if (sz == (num_node*num_dim) * sizeof(VALUETYPE) + sizeof(uint64_t)) {
    fread((void*)b, sizeof(VALUETYPE), num_node*num_dim, fp_mtx2);
  }
  else {
      printf("size of file(%s) is wrong\n", filename);
      return 0;
  }
  fclose(fp_mtx2);

  //printf("b[0] = %f\n" ,b[0]);
  //printf("b[1] = %f\n" ,b[1]);
  //printf("b[2] = %f\n" ,b[2]);
  //printf("b[3] = %f\n" ,b[3]);
  // ===============================================================================//

  // ===============================================================================//
  // pntrb from file
  strcpy(filename, dataset_path);
  strcat(filename, "_pntrb.dat");
  printf("file name = %s\n", filename);
  FILE* fp_mtx3 = fopen(filename, "rb");
  INDEXTYPE* pntrb = (INDEXTYPE*) aligned_alloc(CACHE_LINE_SIZE,  total_num_node * sizeof(INDEXTYPE));

  if (fp_mtx3 != NULL) {
    fseek(fp_mtx3, 0L, SEEK_END);
    uint64_t sz = ftell(fp_mtx3);
    fseek(fp_mtx3, 0L, SEEK_SET);
    if (sz == total_num_node * sizeof(INDEXTYPE)) {
      fread((void*)pntrb, sizeof(INDEXTYPE), total_num_node, fp_mtx3);
    }
    else {
        printf("size of file(%s) is wrong\n", filename);
        return 0;
    }
    fclose(fp_mtx3);
  }
  else {
    printf("Cannot find %s\n", filename);
    return 0;
  }

  //printf("pntrb[0] = %lu\n" ,pntrb[0]);
  //printf("pntrb[1] = %lu\n" ,pntrb[1]);
  //printf("pntrb[2] = %lu\n" ,pntrb[2]);
  //printf("pntrb[3] = %lu\n" ,pntrb[3]);
  // ===============================================================================//

  // ===============================================================================//
  // pntre from file
  strcpy(filename, dataset_path);
  strcat(filename, "_pntre.dat");
  printf("file name = %s\n", filename);
  FILE* fp_mtx4 = fopen(filename, "rb");
  INDEXTYPE* pntre = (INDEXTYPE*) aligned_alloc(CACHE_LINE_SIZE,  total_num_node * sizeof(INDEXTYPE));

  if (fp_mtx4 != NULL) {
    fseek(fp_mtx4, 0L, SEEK_END);
    uint64_t sz = ftell(fp_mtx4);
    fseek(fp_mtx4, 0L, SEEK_SET);
    if (sz == total_num_node * sizeof(INDEXTYPE)) {
      fread((void*)pntre, sizeof(INDEXTYPE), total_num_node, fp_mtx4);
    }
    else {
        printf("size of file(%s) is wrong\n", filename);
        return 0;
    }
    fclose(fp_mtx4);
  }
  else {
    printf("Cannot find %s\n", filename);
    return 0;
  }

  //printf("pntre[0] = %lu\n" ,pntre[0]);
  //printf("pntre[1] = %lu\n" ,pntre[1]);
  //printf("pntre[2] = %lu\n" ,pntre[2]);
  //printf("pntre[3] = %lu\n" ,pntre[3]);
  // ===============================================================================//


  // ===============================================================================//
  // val from file
  strcpy(filename, dataset_path);
  strcat(filename, "_val.dat");
  printf("file name = %s\n", filename);
  FILE* fp_mtx5 = fopen(filename, "rb");
  VALUETYPE* val = (VALUETYPE*) aligned_alloc(CACHE_LINE_SIZE,  nonzero * sizeof(VALUETYPE));

  if (fp_mtx5 != NULL) {
    fseek(fp_mtx5, 0L, SEEK_END);
    uint64_t sz = ftell(fp_mtx5);
    fseek(fp_mtx5, 0L, SEEK_SET);
    if (sz == nonzero * sizeof(VALUETYPE)) {
      fread((void*)val, sizeof(VALUETYPE), nonzero, fp_mtx5);
    }
    else {
        printf("size of file(%s) is wrong\n", filename);
        return 0;
    }
    fclose(fp_mtx5);
  }
  else {
    printf("Cannot find %s\n", filename);
    return 0;
  }

  //printf("val[0] = %f\n" ,val[0]);
  //printf("val[1] = %f\n" ,val[1]);
  //printf("val[2] = %f\n" ,val[2]);
  //printf("val[3] = %f\n" ,val[3]);
  // ===============================================================================//

  // c alloc
  VALUETYPE* c = (VALUETYPE*) aligned_alloc(CACHE_LINE_SIZE,  total_num_node*num_dim * sizeof(VALUETYPE));
  INDEXTYPE ldc = num_dim;

  // m, k
  INDEXTYPE m = total_num_node; // rows of A 
  INDEXTYPE k = num_dim;	    // dimension: col of A and B

  // ===============================================================================//
  // a from file
  strcpy(filename, dataset_path);
  strcat(filename, "_a_mat.dat");
  printf("file name = %s\n", filename);
  FILE* fp_mtx6 = fopen(filename, "rb");
  VALUETYPE* a = (VALUETYPE*) aligned_alloc(CACHE_LINE_SIZE, num_node*num_dim * sizeof(VALUETYPE));
  INDEXTYPE lda = num_dim;

  if (fp_mtx6 != NULL) {
    fseek(fp_mtx6, 0L, SEEK_END);
    uint64_t sz = ftell(fp_mtx6);
    fseek(fp_mtx6, 0L, SEEK_SET);
    printf("sz = %d, num_node = %d, num_dim = %d\n",sz, num_node, num_dim);
    if (sz == num_node*num_dim * sizeof(VALUETYPE)) {
      fread((void*)a, sizeof(VALUETYPE), num_node*num_dim, fp_mtx6);
    }
    else {
        printf("size of file(%s) is wrong\n", filename);
        return 0;
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
    else {
        printf("size of file(%s) is wrong\n", filename);
        return 0;
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

#ifdef WARM_CACHE
#if DIM == 104
    sgfusedMM_K104_sigmoid_b0_csr(64, k, val, indx, pntrb, pntre, a, lda, b, ldb, c, ldc, sm_table);
#elif DIM == 128
    sgfusedMM_K128_sigmoid_b0_csr(64, k, val, indx, pntrb, pntre, a, lda, b, ldb, c, ldc, sm_table);
#elif DIM == 256
    sgfusedMM_K256_sigmoid_b0_csr(64, k, val, indx, pntrb, pntre, a, lda, b, ldb, c, ldc, sm_table);
#elif DIM == 304
    sgfusedMM_K304_sigmoid_b0_csr(64, k, val, indx, pntrb, pntre, a, lda, b, ldb, c, ldc, sm_table);
#elif DIM == 608
    sgfusedMM_K608_sigmoid_b0_csr(64, k, val, indx, pntrb, pntre, a, lda, b, ldb, c, ldc, sm_table);
#endif
#endif

#ifdef GEM_FORGE
  gf_reset_stats();
#endif

  for (uint64_t i = 0; i < num_iter; i++) {
      //truested_spmm_csr(m, k, val, indx, pntrb, pntre, b, ldb, c, ldc);
#if DIM == 104
    sgfusedMM_K104_sigmoid_b0_csr(m, k, val, indx, pntrb, pntre, a, lda, b, ldb, c, ldc, sm_table);
#elif DIM == 128
    sgfusedMM_K128_sigmoid_b0_csr(m, k, val, indx, pntrb, pntre, a, lda, b, ldb, c, ldc, sm_table);
#elif DIM == 256
    sgfusedMM_K256_sigmoid_b0_csr(m, k, val, indx, pntrb, pntre, a, lda, b, ldb, c, ldc, sm_table);
#elif DIM == 304
    sgfusedMM_K304_sigmoid_b0_csr(m, k, val, indx, pntrb, pntre, a, lda, b, ldb, c, ldc, sm_table);
#elif DIM == 608
    sgfusedMM_K608_sigmoid_b0_csr(m, k, val, indx, pntrb, pntre, a, lda, b, ldb, c, ldc, sm_table);
#endif
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
