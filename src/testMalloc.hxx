#pragma once
#include <cstdlib>
#include <cstring>
#include <ctime>

using std::clock_t;
using std::free;
using std::malloc;
using std::memcpy;
using std::clock;


// Testing performance of 100 memory copy operations
// between CPU memory allocated with malloc().
float testMalloc(int size) {
  void *a = malloc(size);
  void *b = malloc(size);
  clock_t start = clock();

  for (int i=0; i<100; i++)
    memcpy(b, a, size);

  clock_t stop = clock();
  float duration = (float) (stop - start) / CLOCKS_PER_SEC;
  return duration * 1000;
}
