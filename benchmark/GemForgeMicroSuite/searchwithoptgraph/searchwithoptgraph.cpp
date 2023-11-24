#ifdef GEM_FORGE
#include "../gfm_utils.h"
#endif


#include <iostream>
#include <algorithm>
#include <vector> 
using namespace std;

//#include <boost/dynamic_bitset.hpp>
//#include <efanna2e/index_nsg.h>
//#include <efanna2e/util.h>

#include <chrono>
#include <string>


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

//#define CHECK

//#define PSP

typedef float Value;

struct Neighbor {
    INDEXTYPE id;
    float distance;
    bool flag;

    Neighbor() = default;
    Neighbor(INDEXTYPE id, float distance, bool f) : id{id}, distance{distance}, flag(f) {}

    inline bool operator<(const Neighbor &other) const {
        return distance < other.distance;
    }
};

static inline int InsertIntoPool (Neighbor *addr, unsigned K, Neighbor nn) {
  // find the location to insert
  int left=0,right=K-1;
  if(addr[left].distance>nn.distance){
    memmove((char *)&addr[left+1], &addr[left],K * sizeof(Neighbor));
    addr[left] = nn;
    return left;
  }
  if(addr[right].distance<nn.distance){
    addr[K] = nn;
    return K;
  }
  while(left<right-1){
    int mid=(left+right)/2;
    if(addr[mid].distance>nn.distance)right=mid;
    else left=mid;
  }
  //check equal ID

  while (left > 0){
    if (addr[left].distance < nn.distance) break;
    if (addr[left].id == nn.id) return K + 1;
    left--;
  }
  if(addr[left].id == nn.id||addr[right].id==nn.id)return K+1;
  memmove((char *)&addr[right+1], &addr[right],(K-right) * sizeof(Neighbor));
  addr[right]=nn;
  return right;
}


// Turn off Cache warm-up by default, because it is not our assumption
//#define WARM_CACHE
__attribute__((noinline)) Value SearchWithOptGraph(
   const INDEXTYPE ep_,        //
   const INDEXTYPE L_para,     //
   const INDEXTYPE K_para,     //
   const INDEXTYPE nd_,        //
   const INDEXTYPE dimension_, //
   std::vector<INDEXTYPE>& flags,
   const VALUETYPE *data_mat,  //
   const VALUETYPE *norm_mat,  //
   VALUETYPE *query_mat, //
   const INDEXTYPE *indx,      // colids -> column indices of sparse matrix 
   const INDEXTYPE *pntrb,     // starting index for rowptr of csr of sparse matrix
   const INDEXTYPE *pntre,     // ending index for rowptr of csr of sparse matrix 
         INDEXTYPE *indices,
         INDEXTYPE *filtered_indx
){

	INDEXTYPE L = L_para;

	// function start!!
	std::vector<Neighbor> retset(L_para + 1);
	//std::vector<Neighbor> retset(L_para);
	std::vector<INDEXTYPE> init_ids(L_para);
	//std::vector<INDEXTYPE> flags(nd_,0);
 	std::fill(flags.begin(), flags.end(), 0);
    //flags.clear();

	INDEXTYPE tmp_l = 0;
	INDEXTYPE MaxM_ep =  pntre[ep_] - pntrb[ep_];	

	printf("MaxM_ep = %d\n", MaxM_ep);

    // modified
    for (INDEXTYPE j=pntrb[ep_]; j < pntre[ep_]; j++) {
       init_ids[tmp_l]= indx[j];
       flags[j] = 1;
       tmp_l++;
    }

	  // original
    //for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
    //  init_ids[tmp_l] = neighbors[tmp_l];
    //  flags[init_ids[tmp_l]] = true;
    //  printf("init_ids_syo[%d] = %d, init_ids[%d] = %d\n",tmp_l,init_ids_syo[tmp_l],tmp_l,init_ids[tmp_l]);
    //}

    while (tmp_l < L) {
      unsigned id = rand() % nd_;
      if (flags[id]) continue;
      flags[id] = 1;
      init_ids[tmp_l] = id;
      tmp_l++;
    }

	//for (unsigned i = 0; i < init_ids.size(); i++) {
  	//  unsigned id = init_ids[i];
  	//  if (id >= nd_) continue;
  	//  _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
  	//}

  	L = 0;
  	for (unsigned i = 0; i < init_ids.size(); i++) {
  	  unsigned id = init_ids[i];
  	  if (id >= nd_) continue;
  	  float norm_x = norm_mat[id]; 
  	  float dist=0;
  	  for(INDEXTYPE ii=0; ii<dimension_; ii=ii+1){
  	  	dist += data_mat[dimension_*id+ii] * query_mat[ii];		
  	  }
  	  dist = -2*dist + norm_x;	  
  	  retset[i] = Neighbor(id, dist, true);
  	  flags[id] = 1;
  	  L++;
  	  //printf("i = %d, id = %d, dist = %f\n", i, id, dist);
  	}

	std::sort(retset.begin(), retset.begin() + L);

	int k = 0;
	while (k < (int)L) {
	  int nk = L;
	
	  if (retset[k].flag) {
	    retset[k].flag = false;
	    INDEXTYPE n = retset[k].id;

      uint64_t filtered_indx_begin = 0;
      uint64_t filtered_indx_end = 0;
      for (unsigned m = pntrb[n]; m < pntre[n]; ++m) {
        unsigned id = indx[m];
        if (flags[id]) continue;
        flags[id] = 1;
        filtered_indx_end++;
        filtered_indx[filtered_indx_end] = id;
      }

#ifdef PSP
      if (filtered_indx_begin < filtered_indx_end) {
        __asm__ volatile (
            "stream.input.offset.begin  $0, %[filtered_indx_begin] \t\n"
            "stream.input.offset.end  $0, %[filtered_indx_end] \t\n"
            "stream.input.ready $0  \t\n"
            :
            :[filtered_indx_begin]"r"(filtered_indx_begin), [filtered_indx_end]"r"(filtered_indx_end)
        );
      }
#endif

//	    for (unsigned m = pntrb[n]; m < pntre[n]; ++m) {
	    for (unsigned m = filtered_indx_begin; m < filtered_indx_end; ++m) {
	      unsigned id = filtered_indx[m];
		  //printf("k = %d, id = %d, n_id=%d\n", k, n, id);
        float norm_x = norm_mat[id];
        float dist=0;
        for(INDEXTYPE ii=0; ii<dimension_; ii=ii+1){
          dist += data_mat[dimension_*id+ii] * query_mat[ii];		
        }
        dist = -2*dist + norm_x;	  

	      if (dist >= retset[L - 1].distance) continue;
	      Neighbor nn(id, dist, true);
	      int r = InsertIntoPool(retset.data(), L, nn);
	
	      // if(L+1 < retset.size()) ++L;
	      if (r < nk) nk = r;
	    }
	  }
	  if (nk <= k)
	    k = nk;
	  else
	    ++k;
	}
	return 0;
}

int main(int argc, char **argv) {
  int numThreads = 1;
  printf("argc: %d, argv[1]: %s, argv[2]: %s, argv[3]: %s, argv[4]: %s, argv[5]: %s, argv[6]: %s\n", argc, argv[1], argv[2], argv[3], argv[4], argv[5], argv[6]);
  if (argc > 1) {
    numThreads = atoi(argv[1]);
  }
  printf("Number of Threads: %d.\n", numThreads);
  omp_set_dynamic(0);
  omp_set_num_threads(numThreads);
  omp_set_schedule(omp_sched_static, 0);
  srand(0);

  char *ptr = NULL;
  char base_path[256] = "";
  char base_name[256] = "";
  char query_path[256] = "";
  char query_name[256] = "";
  char graph_path[256] = "";
  char graph_name[256] = "";
  char filename[100];
  char input_base_path[256] = "";
  char input_query_path[256] = "";
  char input_graph_path[256] = "";

  uint64_t nonzero;
  uint64_t sz;
  uint64_t num_node;
  uint64_t num_query;
  uint64_t total_num_node;
  uint64_t num_dim;
  uint64_t num_query_dim;
  // argv[2] : base
  // argv[3] : query
  // argv[4] : graph
  // argv[5] : L
  // argv[6] : K
  // base_path
  strcpy(input_base_path,argv[2]);
  ptr = strrchr(input_base_path, '/');
  if(ptr==NULL){
  	strcpy(base_name,input_base_path);
  }
  else{
	strcpy(base_name,ptr+1);
  }
  //printf("base_name = %s\n",base_name); 
  strcpy(base_path, input_base_path);
  //printf("base_path = %s\n", base_path);
  strcat(base_path, "/");
  printf("base_path = %s\n", base_path);
  strcat(base_path, base_name);

  // query_path
  strcpy(input_query_path,argv[3]);
  ptr = strrchr(input_query_path, '/');
  if(ptr==NULL){
  	strcpy(query_name,input_query_path);
  }
  else{
	strcpy(query_name,ptr+1);
  }
  //printf("query_name = %s\n",query_name); 
  strcpy(query_path, input_query_path);
  //printf("query_path = %s\n", query_path);
  strcat(query_path, "/");
  printf("query_path = %s\n", query_path);
  strcat(query_path, query_name);

  // graph_path
  strcpy(input_graph_path,argv[4]);
  ptr = strrchr(input_graph_path, '/');
  if(ptr==NULL){
  	strcpy(graph_name,input_graph_path);
  }
  else{
	strcpy(graph_name,ptr+1);
  }
  //printf("graph_name = %s\n",graph_name); 
  strcpy(graph_path, input_graph_path);
  //printf("graph_path = %s\n", graph_path);
  strcat(graph_path, "/");
  printf("graph_path = %s\n", graph_path);
  strcat(graph_path, graph_name);

  INDEXTYPE L = atoi(argv[5]);
  INDEXTYPE K = atoi(argv[6]);
  INDEXTYPE ep_;

  // ===============================================================================//
  // indx from file
  strcpy(filename, graph_path);
  strcat(filename, "_rows.dat");
  printf("file name = %s\n", filename);

  FILE* fp_mtx = fopen(filename, "rb");
  if (fp_mtx != NULL) {
    fseek(fp_mtx, 0L, SEEK_END);
    sz = ftell(fp_mtx);
    fseek(fp_mtx, 0L, SEEK_SET);
	fread((void*)&total_num_node, sizeof(INDEXTYPE), 1, fp_mtx);	
	fread((void*)&nonzero, sizeof(INDEXTYPE), 1, fp_mtx);	
	fread((void*)&ep_, sizeof(INDEXTYPE), 1, fp_mtx);
	printf("total_num_node = %d, nonzero = %d, ep_ =%d\n", total_num_node, nonzero, ep_);
  }
  else {
    printf("Cannot find %s\n", filename);
    return 0;
  }

  //printf("nonzero = %d, total_num_node = %d\n",nonzero, total_num_node);
  INDEXTYPE* indx = (INDEXTYPE*) aligned_alloc(CACHE_LINE_SIZE,  nonzero * sizeof(INDEXTYPE));

  if (sz == (nonzero+3) * sizeof(INDEXTYPE)) {
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
  strcpy(filename, base_path);
  strcat(filename, "_b_mat.dat");
  printf("file name = %s\n", filename);
  FILE* fp_mtx2 = fopen(filename, "rb");
  if (fp_mtx2 != NULL) {
    fseek(fp_mtx2, 0L, SEEK_END);
    sz = ftell(fp_mtx2);	
    fseek(fp_mtx2, 0L, SEEK_SET);
	fread((void*)&num_dim, sizeof(uint64_t), 1, fp_mtx2);	
  }
  else {
    printf("Cannot find %s\n", filename);
    return 0;
  }

  num_node = (sz-sizeof(uint64_t))/(sizeof(VALUETYPE)*num_dim);
  printf("num_dim = %d, num_node = %d\n", sz, num_dim, num_node);
  VALUETYPE* b = (VALUETYPE*) aligned_alloc(CACHE_LINE_SIZE,  num_node*num_dim * sizeof(VALUETYPE));
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
  strcpy(filename, graph_path);
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
  strcpy(filename, graph_path);
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

  // ==============================================================================//
  // query from file
  strcpy(filename, query_path);
  strcat(filename, "_query.dat");
  printf("file name = %s\n", filename);
  FILE* fp_mtx5 = fopen(filename, "rb");
  if (fp_mtx5 != NULL) {
    fseek(fp_mtx5, 0L, SEEK_END);
    sz = ftell(fp_mtx5);	
    fseek(fp_mtx5, 0L, SEEK_SET);
	fread((void*)&num_query_dim, sizeof(uint64_t), 1, fp_mtx5);	
    //printf("sz = %d, num_dim = %d\n", sz, num_dim);
  }
  else {
    printf("Cannot find %s\n", filename);
    return 0;
  }

  num_query = (sz-sizeof(uint64_t))/(sizeof(VALUETYPE)*num_query_dim);
  //printf("sz = %d, num_dim = %d, num_node = %d\n", sz, num_dim, num_node);
  VALUETYPE* query = (VALUETYPE*) aligned_alloc(CACHE_LINE_SIZE,  num_query*num_query_dim * sizeof(VALUETYPE));
  if (sz == (num_query*num_dim) * sizeof(VALUETYPE) + sizeof(uint64_t)) {
    fread((void*)query, sizeof(VALUETYPE), num_query*num_query_dim, fp_mtx5);
  }
  else {
      printf("size of file(%s) is wrong\n", filename);
      return 0;
  }
  fclose(fp_mtx5);

  //printf("query[0] = %f\n" ,query[0]);
  //printf("query[1] = %f\n" ,query[1]);
  //printf("query[2] = %f\n" ,query[2]);
  //printf("query[3] = %f\n" ,query[3]);
  // ===============================================================================//
  if (num_dim != num_query_dim){
	  printf("num_dim(%d) != num_query_dim(%d)\n", num_dim, num_query_dim);
	  return 0;
  }

  // ===============================================================================//
  // norm from file
  strcpy(filename, base_path);
  strcat(filename, "_norm.dat");
  printf("file name = %s\n", filename);
  FILE* fp_mtx6 = fopen(filename, "rb");
  VALUETYPE* norm = (VALUETYPE*) aligned_alloc(CACHE_LINE_SIZE,  total_num_node * sizeof(VALUETYPE));

  if (fp_mtx6 != NULL) {
    fseek(fp_mtx6, 0L, SEEK_END);
    uint64_t sz = ftell(fp_mtx6);
    fseek(fp_mtx6, 0L, SEEK_SET);
    if (sz == total_num_node * sizeof(VALUETYPE)) {
      fread((void*)norm, sizeof(VALUETYPE), total_num_node, fp_mtx6);
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

  //printf("norm[0] = %lu\n" ,norm[0]);
  //printf("norm[1] = %lu\n" ,norm[1]);
  //printf("norm[2] = %lu\n" ,norm[2]);
  //printf("norm[3] = %lu\n" ,norm[3]);
  // ===============================================================================//
#ifdef GEM_FORGE
  gf_detail_sim_start();
#endif

#ifdef GEM_FORGE
  gf_reset_stats();
#endif

  std::vector<INDEXTYPE*> filtered_indx(numThreads);
  for (uint64_t i = 0; i < numThreads; i++) {
    filtered_indx[i] = (INDEXTYPE*)aligned_alloc(CACHE_LINE_SIZE, 100 * sizeof(INDEXTYPE));
    memset(filtered_indx[i], 0, 100 * sizeof(INDEXTYPE));
  }

#pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < numThreads; i++) {
    INDEXTYPE* idx_base_addr = filtered_indx[i];
    uint64_t idx_granularity = sizeof(INDEXTYPE);
    VALUETYPE* val_base_addr = b;
    uint64_t val_granularity = num_dim * sizeof(VALUETYPE);
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
  std::vector<std::vector<INDEXTYPE>> res(num_query);
  for (INDEXTYPE i = 0; i < num_query; i++) res[i].resize(K);

  std::vector<INDEXTYPE> flags(num_node,0);
#pragma omp parallel for schedule(static)
  for (INDEXTYPE i = 0; i < num_query; i++) {
	  SearchWithOptGraph(ep_, L, K, num_node, num_dim, flags, b, norm, query + i * num_dim, indx, pntrb, pntre, res[i].data(), filtered_indx[i]);
  }

#ifdef PSP
   #pragma omp parallel for schedule(static)
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

  for (uint64_t i = 0; i < numThreads; i++) {
    free(filtered_indx[i]);
  }
  free(pntrb);
  free(pntre);
  free(indx);
  free(b);
  free(query);
  free(norm);

  return 0;
}
