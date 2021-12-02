malloc() is a function for dynamic memory allocation in C.

```c
testMalloc():
Testing performance of 100 memory copy operations
between CPU memory allocated with malloc().
```

```c
testCudaMalloc():
Testing performance of 100 memory copy operations
between CPU memory allocated with malloc() and
GPU memory allocated with cudaMalloc(). Because
memory allocated with malloc() is pageable memory,
it will first be copied to a page-locked `staging`
area, before being transferring to GPU by DMA.
Note however that allocating too much pinned memory
can cause system slowdown, or even crash due to
lack of usable memory.
```

```c
testCudaHostAlloc():
Testing performance of 100 memory copy operations
between CPU memory allocated with cudaHostAlloc()
and GPU memory allocated with cudaMalloc(). Memory
allocated with cudaHostAlloc() is page-locked
(pinned), which means the memory can be directly
copied by DMA into the GPU.
```

```bash
$ nvcc -std=c++17 -Xcompiler -O3 main.cu
$ ./a.out

# CPU malloc -> CPU malloc: 0.0 ms
#
# CPU malloc -> GPU cudaMalloc: 208.9 ms
# CPU malloc <- GPU cudaMalloc: 178.7 ms
#
# CPU cudaHostAlloc -> GPU cudaMalloc: 86.1 ms
# CPU cudaHostAlloc <- GPU cudaMalloc: 80.4 ms
```

See [main.cu] for code.

[main.cu]: main.cu

<br>
<br>


## References

- [CUDA by Example :: Jason Sanders, Edward Kandrot](https://gist.github.com/wolfram77/72c51e494eaaea1c21a9c4021ad0f320)
