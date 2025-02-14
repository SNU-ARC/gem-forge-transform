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
#include <time.h>
#include <sys/mman.h>
#include <fcntl.h>


#include <omp.h>
#include "immintrin.h"


// yosong, 231016
#define INDEXTYPE uint64_t
#define VALUETYPE float

static const uint64_t num_iter			  = 1;
//#define CHECK

#define PTTIME
//#define PSP

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

     //printf("&b: %x, indx: %lu, offset_begin: %lu, offset_end: %lu.\n", &b[indx[j] * ldb], indx[j], j, offset_end);
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

  char *ptr = NULL;
  char dataset_path[256] = "";
  char dataset_name[256] = "";
  char filename[400];
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
    printf("sz = %lu, num_dim = %d\n", sz, num_dim);
  }
  else {
    printf("Cannot find %s\n", filename);
    return 0;
  }

  num_node = (sz-sizeof(uint64_t))/(sizeof(VALUETYPE)*num_dim);
  INDEXTYPE ldb = num_dim;
  //printf("sz = %d, num_dim = %d, num_node = %d\n", sz, num_dim, num_node);
  VALUETYPE* b;
  int fp_mtx2_mmap = open(filename, O_RDONLY);
  if (fp_mtx2_mmap != -1) {
    b = (VALUETYPE*)mmap(0, num_node * num_dim * sizeof(VALUETYPE) + sizeof(uint64_t), PROT_READ, MAP_SHARED, fp_mtx2_mmap, 0);
    b += sizeof(uint64_t);
  }
//  VALUETYPE* b = (VALUETYPE*) aligned_alloc(CACHE_LINE_SIZE,  num_node*num_dim * sizeof(VALUETYPE));
//  if (sz == (num_node*num_dim) * sizeof(VALUETYPE) + sizeof(uint64_t)) {
//    fread((void*)b, sizeof(VALUETYPE), num_node*num_dim, fp_mtx2);
//  }
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
  printf("%lu %lu\n", nonzero, sizeof(VALUETYPE));
  VALUETYPE* val;
//  VALUETYPE* val = (VALUETYPE*) aligned_alloc(CACHE_LINE_SIZE,  nonzero * sizeof(VALUETYPE));
  
  // val from file
  strcpy(filename, dataset_path);
  strcat(filename, "_val.dat");
  printf("file name = %s\n", filename);
//  FILE* fp_mtx5 = fopen(filename, "rb");
//  if (fp_mtx5 != NULL) {
//    fseek(fp_mtx5, 0L, SEEK_END);
//    uint64_t sz = ftell(fp_mtx5);
//    fseek(fp_mtx5, 0L, SEEK_SET);
//    if (sz == nonzero * sizeof(VALUETYPE)) {
//      fread((void*)val, sizeof(VALUETYPE), nonzero, fp_mtx5);
//    }
//    else {
//      printf("size of file(%s) is wrong\n", filename);
//      return 0;
//    }
//    fclose(fp_mtx5);
//  }

  int fp_mtx5 = open(filename, O_RDONLY);

  if (fp_mtx5 != -1) {
//      read(fp_mtx5, val, nonzero * sizeof(VALUETYPE));
      val = (VALUETYPE*)mmap(0, nonzero * sizeof(VALUETYPE), PROT_READ, MAP_SHARED, fp_mtx5, 0);
      printf("size of val is %lu %lu %#x %d\n", sz, sizeof(VALUETYPE) * nonzero, val, fp_mtx5);
      printf("size of val is %lu\n", sizeof(VALUETYPE) * nonzero);
      close(fp_mtx5);
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
  printf("size of C is %lu %lu %lu\n", total_num_node, num_dim, sizeof(VALUETYPE));
  VALUETYPE* c = (VALUETYPE*) aligned_alloc(CACHE_LINE_SIZE,  total_num_node*num_dim * sizeof(VALUETYPE));
  INDEXTYPE ldc = num_dim;

  // m, k
  INDEXTYPE m = total_num_node; // rows of A 
  INDEXTYPE k = num_dim;	    // dimension: col of A and B

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
  // ===============================================================================//
  // _spmm_c_mat from file
  strcpy(filename, dataset_path);
  strcat(filename, "_spmm_c_mat.dat");
  printf("file name = %s\n", filename);
  FILE* fp_mtx6 = fopen(filename, "rb");
  VALUETYPE* c_golden = (VALUETYPE*) aligned_alloc(CACHE_LINE_SIZE, total_num_node*num_dim * sizeof(VALUETYPE));

  if (fp_mtx6 != NULL) {
    fseek(fp_mtx6, 0L, SEEK_END);
    sz = ftell(fp_mtx6);
    fseek(fp_mtx6, 0L, SEEK_SET);
    //printf("sz = %lu, sizeof(sz) = %d, num_node*num_dim * sizeof(VALUETYPE) = %lu\n", sz, sizeof(sz), num_node*num_dim * sizeof(VALUETYPE));
    if (sz == num_node * num_dim * sizeof(VALUETYPE)) {
        fread((void*)c_golden, sizeof(VALUETYPE), num_node*num_dim, fp_mtx6);
    }
    else {
        printf("size of file(%s) is wrong\n", filename);
        return 0;
    }
    fclose(fp_mtx6);	
  }
  else {
    printf("Cannot find %s\n", filename);
    return 0;
  }

  if( memcmp(c_golden, c, total_num_node*num_dim * sizeof(VALUETYPE)) == 0 )
  {
	printf("PPPPPPPPPPPPPPPPPPPPPPPPPPPP\n");
    printf("Matched:: c_golden and c\n");
	printf("PPPPPPPPPPPPPPPPPPPPPPPPPPPP\n");
  } 
  else{
	printf("FFFFFFFFFFFFFFFFFFFFFFFFFFFF\n");
    printf("unmatched:: c_golden and c\n");
	printf("FFFFFFFFFFFFFFFFFFFFFFFFFFFF\n");
  }


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
  free(c_golden);
  //free(C1);
#endif

//  free(val);
  munmap(val, sizeof(VALUETYPE) * nonzero);
  free(indx);
  free(pntrb);
  free(pntre);
//  free(b);
  free(c);

  return 0;
}
