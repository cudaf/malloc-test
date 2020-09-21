#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "support.h"


// Testing performance of 100 memory copy operations
// between CPU memory allocated with malloc().
float test_malloc(int size) {
  void *a = malloc(size);
  void *b = malloc(size);
  clock_t start = clock();

  for (int i=0; i<100; i++)
    memcpy(b, a, size);

  clock_t stop = clock();
  float duration = (float)(stop - start) / CLOCKS_PER_SEC;
  return duration * 1000;
}


// Testing performance of 100 memory copy operations
// between CPU memory allocated with malloc() and
// GPU memory allocated with cudaMalloc(). Because
// memory allocated with malloc() is pageable memory,
// it will first be copied to a page-locked `staging`
// area, before being transferring to GPU by DMA.
// Note however that allocating too much pinned memory
// can cause system slowdown, or even crash due to
// lack of usable memory.
float test_cuda_malloc(int size, bool up) {
  cudaEvent_t start, stop;
  TRY( cudaEventCreate(&start) );
  TRY( cudaEventCreate(&stop) );

  void *a, *aD;
  a = malloc(size);
  TRY( cudaMalloc(&aD, size) );
  TRY( cudaEventRecord(start, 0) );

  for (int i=0; i<100; i++) {
    if (up) TRY( cudaMemcpy(aD, a, size, cudaMemcpyHostToDevice) );
    else TRY( cudaMemcpy(a, aD, size, cudaMemcpyDeviceToHost) );
  }

  float duration;
  TRY( cudaEventRecord(stop, 0) );
  TRY( cudaEventSynchronize(stop) );
  TRY( cudaEventElapsedTime(&duration, start, stop) );

  TRY( cudaEventDestroy(start) );
  TRY( cudaEventDestroy(stop) );
  TRY( cudaFree(aD) );
  free(a);
  return duration;
}


// Testing performance of 100 memory copy operations
// between CPU memory allocated with cudaHostAlloc()
// and GPU memory allocated with cudaMalloc(). Memory
// allocated with cudaHostAlloc() is page-locked
// (pinned), which means the memory can be directly
// copied by DMA into the GPU.
float test_cuda_host_alloc(int size, bool up) {
  cudaEvent_t start, stop;
  TRY( cudaEventCreate(&start) );
  TRY( cudaEventCreate(&stop) );

  void *a, *aD;
  TRY( cudaHostAlloc(&a, size, cudaHostAllocDefault) );
  TRY( cudaMalloc(&aD, size) );
  TRY( cudaEventRecord(start, 0) );

  for (int i=0; i<100; i++) {
    if (up) TRY( cudaMemcpy(aD, a, size, cudaMemcpyHostToDevice) );
    else TRY( cudaMemcpy(a, aD, size, cudaMemcpyDeviceToHost) );
  }

  float duration;
  TRY( cudaEventRecord(stop, 0) );
  TRY( cudaEventSynchronize(stop) );
  TRY( cudaEventElapsedTime(&duration, start, stop) );

  TRY( cudaEventDestroy(start) );
  TRY( cudaEventDestroy(stop) );
  TRY( cudaFree(aD) );
  TRY( cudaFreeHost(a) );
  return duration;
}


int main() {
  int size = 10 * 1024 * 1024;

  printf("CPU malloc -> CPU malloc: %3.1f ms\n",
    test_malloc(size));
  printf("\n");

  printf("CPU malloc -> GPU cudaMalloc: %3.1f ms\n",
    test_cuda_malloc(size, 1));
  printf("CPU malloc <- GPU cudaMalloc: %3.1f ms\n",
    test_cuda_malloc(size, 0));
  printf("\n");

  printf("CPU cudaHostAlloc -> GPU cudaMalloc: %3.1f ms\n",
    test_cuda_host_alloc(size, 1));
  printf("CPU cudaHostAlloc <- GPU cudaMalloc: %3.1f ms\n",
    test_cuda_host_alloc(size, 0));
  return 0;
}
