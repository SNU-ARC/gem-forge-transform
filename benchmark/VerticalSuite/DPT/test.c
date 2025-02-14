#include <stdio.h>
#include "randArr.h"
#include "common.h"
#include "math.h"

#define ASIZE  1024
#define STEP    128
#define ITERS     4

float arrA[ASIZE];
float arrB[ASIZE];

__attribute__ ((noinline)) 
float loop(int zero) {
  int i, iters;
  float t1;

  for(iters=zero; iters < ITERS; iters+=1) {
    for(i=zero; i < ASIZE; i+=1) {
      arrA[i]=sin(arrA[i]);
    }
    t1+=arrA[ASIZE-1];
  }

  return t1;
}

int main(int argc, char* argv[]) {
   argc&=10000;
   m5_detail_sim_start(); 
   int t=loop(argc); 
   m5_detail_sim_end();
   volatile float a = t;
}
