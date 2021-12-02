#include <cstdio>
#include "src/main.hxx"

using std::printf;


int main() {
  int size = 10 * 1024 * 1024;

  printf("CPU malloc -> CPU malloc: %3.1f ms\n",
    testMalloc(size));
  printf("\n");

  printf("CPU malloc -> GPU cudaMalloc: %3.1f ms\n",
    testCudaMalloc(size, 1));
  printf("CPU malloc <- GPU cudaMalloc: %3.1f ms\n",
    testCudaMalloc(size, 0));
  printf("\n");

  printf("CPU cudaHostAlloc -> GPU cudaMalloc: %3.1f ms\n",
    testCudaHostAlloc(size, 1));
  printf("CPU cudaHostAlloc <- GPU cudaMalloc: %3.1f ms\n",
    testCudaHostAlloc(size, 0));
  printf("\n");
  return 0;
}
