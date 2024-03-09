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
#if VLEN != 8
   #error "ARCH VLEN doesn't match with generator's VLEN, see simd.h " 
#endif

#define BETA0
/*
 * Register block  A,C and B(innermost loop) will require most registers, works 
 * better on small value of k
 */
#if DIM == 104
#ifdef BETA0 
__attribute__((noinline)) Value sgfusedMM_K104_spmm_b0_csr
#else /* BETA1 version */
__attribute__((noinline)) Value sgfusedMM_K104_spmm_b1_csr
#endif
(
   const INDEXTYPE m,      // rows of dense A matrix 
   const INDEXTYPE k,      // cols of A or dimension. not used since K compile time   
   const float *val,       // value of  the sparse matrix 
   const INDEXTYPE *indx,  // colids -> column indices of sparse matrix 
   const INDEXTYPE *pntrb, // starting index for rowptr of csr of sparse matrix
   const INDEXTYPE *pntre, // ending index for rowptr of csr of sparse matrix 
   const float *b,        // Dense B matrix
   const INDEXTYPE ldb,    // leading dimension of b (col size since row-major)  
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
                     Vc11, Va12, Vc12;
      INDEXTYPE iindex = i * 104; 
//      const float *Ai = a + iindex; 
      float *Ci = c + iindex; 
      VTYPE VMAXBOUND, VMINBOUND; 
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
      for (INDEXTYPE j = pntrb[i]; j < pntre[i]; j++)
      {
         VTYPE Vb0, Vb1, Vb2, Vb3, Vb4, Vb5, Vb6, Vb7, Vb8, Vb9, Vb10, Vb11,
               Vb12;
         VTYPE Va0; 
         float a0 = val[j];
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
         BCL_vset1(Va0, a0);
         // spmm vmac 
         BCL_vmac(Vc0, Va0, Vb0);
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
   return 0;
}
#elif DIM == 128
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
#elif DIM == 256
#ifdef BETA0 
__attribute__((noinline)) Value sgfusedMM_K256_spmm_b0_csr
#else /* BETA1 version */
__attribute__((noinline)) Value sgfusedMM_K256_spmm_b1_csr
#endif
(
   const INDEXTYPE m,      // rows of dense A matrix 
   const INDEXTYPE k,      // cols of A or dimension. not used since K compile time   
   const float *val,       // value of  the sparse matrix 
   const INDEXTYPE *indx,  // colids -> column indices of sparse matrix 
   const INDEXTYPE *pntrb, // starting index for rowptr of csr of sparse matrix
   const INDEXTYPE *pntre, // ending index for rowptr of csr of sparse matrix 
   const float *b,        // Dense B matrix
   const INDEXTYPE ldb,    // leading dimension of b (col size since row-major)  
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
                     Vc11, Va12, Vc12, Va13, Vc13, Va14, Vc14, Va15, Vc15, Va16,
                     Vc16, Va17, Vc17, Va18, Vc18, Va19, Vc19, Va20, Vc20, Va21,
                     Vc21, Va22, Vc22, Va23, Vc23, Va24, Vc24, Va25, Vc25, Va26,
                     Vc26, Va27, Vc27, Va28, Vc28, Va29, Vc29, Va30, Vc30, Va31,
                     Vc31;
      INDEXTYPE iindex = i * 256; 
//      const float *Ai = a + iindex; 
      float *Ci = c + iindex; 
      VTYPE VMAXBOUND, VMINBOUND; 
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
      for (INDEXTYPE j = pntrb[i]; j < pntre[i]; j++)
      {
         VTYPE Vb0, Vb1, Vb2, Vb3, Vb4, Vb5, Vb6, Vb7, Vb8, Vb9, Vb10, Vb11,
               Vb12, Vb13, Vb14, Vb15, Vb16, Vb17, Vb18, Vb19, Vb20, Vb21, Vb22,
               Vb23, Vb24, Vb25, Vb26, Vb27, Vb28, Vb29, Vb30, Vb31;
         VTYPE Va0; 
         float a0 = val[j];
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
         BCL_vset1(Va0, a0);
         // spmm vmac 
         BCL_vmac(Vc0, Va0, Vb0);
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
         BCL_vmac(Vc16, Va0, Vb16);
         BCL_vmac(Vc17, Va0, Vb17);
         BCL_vmac(Vc18, Va0, Vb18);
         BCL_vmac(Vc19, Va0, Vb19);
         BCL_vmac(Vc20, Va0, Vb20);
         BCL_vmac(Vc21, Va0, Vb21);
         BCL_vmac(Vc22, Va0, Vb22);
         BCL_vmac(Vc23, Va0, Vb23);
         BCL_vmac(Vc24, Va0, Vb24);
         BCL_vmac(Vc25, Va0, Vb25);
         BCL_vmac(Vc26, Va0, Vb26);
         BCL_vmac(Vc27, Va0, Vb27);
         BCL_vmac(Vc28, Va0, Vb28);
         BCL_vmac(Vc29, Va0, Vb29);
         BCL_vmac(Vc30, Va0, Vb30);
         BCL_vmac(Vc31, Va0, Vb31);
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
   return 0;
}
#elif DIM == 304
#ifdef BETA0 
__attribute__((noinline)) Value sgfusedMM_K304_spmm_b0_csr
#else /* BETA1 version */
__attribute__((noinline)) Value sgfusedMM_K304_spmm_b1_csr
#endif
(
   const INDEXTYPE m,      // rows of dense A matrix 
   const INDEXTYPE k,      // cols of A or dimension. not used since K compile time   
   const float *val,       // value of  the sparse matrix 
   const INDEXTYPE *indx,  // colids -> column indices of sparse matrix 
   const INDEXTYPE *pntrb, // starting index for rowptr of csr of sparse matrix
   const INDEXTYPE *pntre, // ending index for rowptr of csr of sparse matrix 
   const float *b,        // Dense B matrix
   const INDEXTYPE ldb,    // leading dimension of b (col size since row-major)  
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
                     Vc11, Va12, Vc12, Va13, Vc13, Va14, Vc14, Va15, Vc15, Va16,
                     Vc16, Va17, Vc17, Va18, Vc18, Va19, Vc19, Va20, Vc20, Va21,
                     Vc21, Va22, Vc22, Va23, Vc23, Va24, Vc24, Va25, Vc25, Va26,
                     Vc26, Va27, Vc27, Va28, Vc28, Va29, Vc29, Va30, Vc30, Va31,
                     Vc31, Va32, Vc32, Va33, Vc33, Va34, Vc34, Va35, Vc35, Va36,
                     Vc36, Va37, Vc37;
      INDEXTYPE iindex = i * 304; 
//      const float *Ai = a + iindex; 
      float *Ci = c + iindex; 
      VTYPE VMAXBOUND, VMINBOUND; 
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
      for (INDEXTYPE j = pntrb[i]; j < pntre[i]; j++)
      {
         VTYPE Vb0, Vb1, Vb2, Vb3, Vb4, Vb5, Vb6, Vb7, Vb8, Vb9, Vb10, Vb11,
               Vb12, Vb13, Vb14, Vb15, Vb16, Vb17, Vb18, Vb19, Vb20, Vb21, Vb22,
               Vb23, Vb24, Vb25, Vb26, Vb27, Vb28, Vb29, Vb30, Vb31, Vb32, Vb33,
               Vb34, Vb35, Vb36, Vb37;
         VTYPE Va0; 
         float a0 = val[j];
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
         BCL_vset1(Va0, a0);
         // spmm vmac 
         BCL_vmac(Vc0, Va0, Vb0);
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
         BCL_vmac(Vc16, Va0, Vb16);
         BCL_vmac(Vc17, Va0, Vb17);
         BCL_vmac(Vc18, Va0, Vb18);
         BCL_vmac(Vc19, Va0, Vb19);
         BCL_vmac(Vc20, Va0, Vb20);
         BCL_vmac(Vc21, Va0, Vb21);
         BCL_vmac(Vc22, Va0, Vb22);
         BCL_vmac(Vc23, Va0, Vb23);
         BCL_vmac(Vc24, Va0, Vb24);
         BCL_vmac(Vc25, Va0, Vb25);
         BCL_vmac(Vc26, Va0, Vb26);
         BCL_vmac(Vc27, Va0, Vb27);
         BCL_vmac(Vc28, Va0, Vb28);
         BCL_vmac(Vc29, Va0, Vb29);
         BCL_vmac(Vc30, Va0, Vb30);
         BCL_vmac(Vc31, Va0, Vb31);
         BCL_vmac(Vc32, Va0, Vb32);
         BCL_vmac(Vc33, Va0, Vb33);
         BCL_vmac(Vc34, Va0, Vb34);
         BCL_vmac(Vc35, Va0, Vb35);
         BCL_vmac(Vc36, Va0, Vb36);
         BCL_vmac(Vc37, Va0, Vb37);
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
   return 0;
}
#elif DIM == 608
#ifdef BETA0 
__attribute__((noinline)) Value sgfusedMM_K608_spmm_b0_csr
#else /* BETA1 version */
__attribute__((noinline)) Value sgfusedMM_K608_spmm_b1_csr
#endif
(
   const INDEXTYPE m,      // rows of dense A matrix 
   const INDEXTYPE k,      // cols of A or dimension. not used since K compile time   
   const float *val,       // value of  the sparse matrix 
   const INDEXTYPE *indx,  // colids -> column indices of sparse matrix 
   const INDEXTYPE *pntrb, // starting index for rowptr of csr of sparse matrix
   const INDEXTYPE *pntre, // ending index for rowptr of csr of sparse matrix 
   const float *b,        // Dense B matrix
   const INDEXTYPE ldb,    // leading dimension of b (col size since row-major)  
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
//      const float *Ai = a + iindex; 
      float *Ci = c + iindex; 
      VTYPE VMAXBOUND, VMINBOUND; 
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
      for (INDEXTYPE j = pntrb[i]; j < pntre[i]; j++)
      {
         VTYPE Vb0, Vb1, Vb2, Vb3, Vb4, Vb5, Vb6, Vb7, Vb8, Vb9, Vb10, Vb11,
               Vb12, Vb13, Vb14, Vb15, Vb16, Vb17, Vb18, Vb19, Vb20, Vb21, Vb22,
               Vb23, Vb24, Vb25, Vb26, Vb27, Vb28, Vb29, Vb30, Vb31, Vb32, Vb33,
               Vb34, Vb35, Vb36, Vb37, Vb38, Vb39, Vb40, Vb41, Vb42, Vb43, Vb44,
               Vb45, Vb46, Vb47, Vb48, Vb49, Vb50, Vb51, Vb52, Vb53, Vb54, Vb55,
               Vb56, Vb57, Vb58, Vb59, Vb60, Vb61, Vb62, Vb63, Vb64, Vb65, Vb66,
               Vb67, Vb68, Vb69, Vb70, Vb71, Vb72, Vb73, Vb74, Vb75;
         VTYPE Va0; 
         float a0 = val[j];
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
         BCL_vset1(Va0, a0);
         // spmm vmac 
         BCL_vmac(Vc0, Va0, Vb0);
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
         BCL_vmac(Vc16, Va0, Vb16);
         BCL_vmac(Vc17, Va0, Vb17);
         BCL_vmac(Vc18, Va0, Vb18);
         BCL_vmac(Vc19, Va0, Vb19);
         BCL_vmac(Vc20, Va0, Vb20);
         BCL_vmac(Vc21, Va0, Vb21);
         BCL_vmac(Vc22, Va0, Vb22);
         BCL_vmac(Vc23, Va0, Vb23);
         BCL_vmac(Vc24, Va0, Vb24);
         BCL_vmac(Vc25, Va0, Vb25);
         BCL_vmac(Vc26, Va0, Vb26);
         BCL_vmac(Vc27, Va0, Vb27);
         BCL_vmac(Vc28, Va0, Vb28);
         BCL_vmac(Vc29, Va0, Vb29);
         BCL_vmac(Vc30, Va0, Vb30);
         BCL_vmac(Vc31, Va0, Vb31);
         BCL_vmac(Vc32, Va0, Vb32);
         BCL_vmac(Vc33, Va0, Vb33);
         BCL_vmac(Vc34, Va0, Vb34);
         BCL_vmac(Vc35, Va0, Vb35);
         BCL_vmac(Vc36, Va0, Vb36);
         BCL_vmac(Vc37, Va0, Vb37);
         BCL_vmac(Vc38, Va0, Vb38);
         BCL_vmac(Vc39, Va0, Vb39);
         BCL_vmac(Vc40, Va0, Vb40);
         BCL_vmac(Vc41, Va0, Vb41);
         BCL_vmac(Vc42, Va0, Vb42);
         BCL_vmac(Vc43, Va0, Vb43);
         BCL_vmac(Vc44, Va0, Vb44);
         BCL_vmac(Vc45, Va0, Vb45);
         BCL_vmac(Vc46, Va0, Vb46);
         BCL_vmac(Vc47, Va0, Vb47);
         BCL_vmac(Vc48, Va0, Vb48);
         BCL_vmac(Vc49, Va0, Vb49);
         BCL_vmac(Vc50, Va0, Vb50);
         BCL_vmac(Vc51, Va0, Vb51);
         BCL_vmac(Vc52, Va0, Vb52);
         BCL_vmac(Vc53, Va0, Vb53);
         BCL_vmac(Vc54, Va0, Vb54);
         BCL_vmac(Vc55, Va0, Vb55);
         BCL_vmac(Vc56, Va0, Vb56);
         BCL_vmac(Vc57, Va0, Vb57);
         BCL_vmac(Vc58, Va0, Vb58);
         BCL_vmac(Vc59, Va0, Vb59);
         BCL_vmac(Vc60, Va0, Vb60);
         BCL_vmac(Vc61, Va0, Vb61);
         BCL_vmac(Vc62, Va0, Vb62);
         BCL_vmac(Vc63, Va0, Vb63);
         BCL_vmac(Vc64, Va0, Vb64);
         BCL_vmac(Vc65, Va0, Vb65);
         BCL_vmac(Vc66, Va0, Vb66);
         BCL_vmac(Vc67, Va0, Vb67);
         BCL_vmac(Vc68, Va0, Vb68);
         BCL_vmac(Vc69, Va0, Vb69);
         BCL_vmac(Vc70, Va0, Vb70);
         BCL_vmac(Vc71, Va0, Vb71);
         BCL_vmac(Vc72, Va0, Vb72);
         BCL_vmac(Vc73, Va0, Vb73);
         BCL_vmac(Vc74, Va0, Vb74);
         BCL_vmac(Vc75, Va0, Vb75);
         // rolled loop for remaining computation
         for (INDEXTYPE kk=608; kk < k; kk++)
            Ci[kk] +=  a0 * Bj[kk];   
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
  // val alloc
  VALUETYPE* val = (VALUETYPE*) aligned_alloc(CACHE_LINE_SIZE,  nonzero * sizeof(VALUETYPE));
  
  // val from file
  strcpy(filename, dataset_path);
  strcat(filename, "_val.dat");
  printf("file name = %s\n", filename);
  FILE* fp_mtx5 = fopen(filename, "rb");

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
#if DIM == 104
	  sgfusedMM_K104_spmm_b0_csr(m, k, val, indx, pntrb, pntre, b, ldb, c, ldc);
#elif DIM == 128
	  sgfusedMM_K128_spmm_b0_csr(m, k, val, indx, pntrb, pntre, b, ldb, c, ldc);
#elif DIM == 256
	  sgfusedMM_K256_spmm_b0_csr(m, k, val, indx, pntrb, pntre, b, ldb, c, ldc);
#elif DIM == 304
	  sgfusedMM_K304_spmm_b0_csr(m, k, val, indx, pntrb, pntre, b, ldb, c, ldc);
#elif DIM == 608
	  sgfusedMM_K608_spmm_b0_csr(m, k, val, indx, pntrb, pntre, b, ldb, c, ldc);
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
  free(b);
  free(c);

  return 0;
}
