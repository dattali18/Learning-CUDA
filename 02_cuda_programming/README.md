# CUDA Programming Model

In this unit, we will explore further into programming with CUDA, focusing mainly on the application in the world of AI & DL.

## Table of Contents

- [Introduction](#21-introduction)
- [CUDA Programming Model](#22-cuda-programming-model)
 - [CUDA Kernels](#23-kernels)
 - [Thread Hierarchies / Indexing](#24-thread-hierarchies--indexing)
 - [Memory Hierarchy](#25-memory-hierarchy)
 - [Heterogeneous Programming](#26-heterogeneous-programming)

## 2.1 Introduction

CUDA is an extension of the C++ language.

Let's give some simple example of a vector addition operation:

```c
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vector_add(int * a, int * b, int * c, int n) {
    int i = threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

As you can see in the code above in the CUDA programming model, we try and use the GPU to parallelize the vector addition operation. The kernel function `vector_add` is executed by many threads in parallel. Each thread is responsible for adding one element of the array.

## 2.2 CUDA Programming Model

The CUDA programming model is based on the following concepts:

1. **Memory Communication**: The host and device have separate memory spaces. Data must be copied between the host and device memory.
2. **Kernels**: Kernels are functions that are executed on the device. They are executed by many threads in parallel.
3. **Threads**: Threads are organized into blocks.
4. **Blocks**: Blocks are organized into a grid.
5. **Grid**: A grid is a group of blocks.

### 2.2.1 CUDA A Scalable Parallel Programming Model

CUDA is a scalable parallel programming model that allows you to write parallel code that can be executed on a GPU. The abstraction of the specification of the GPU allows us to write generic code and CUDA will try and optimize it for the specific GPU it is running on.

![Automatic Scaling](/images/04_image.png)

As we can see in the illustration above some GPUs have more or less SMs, and CUDA will try and make the most of the available resources.

### 2.2.2 CUDA Thread Hierarchy

GPUs have a different thread hierarchy than CPUs. They have blocks, warps, threads, etc.

Here is an illustration of the thread hierarchy in CUDA:

![CUDA Thread Hierarchy](/images/05_image.png)

As we can see each grid can have many blocks, each block can have many threads. The threads are organized into warps and the warps are organized into blocks. The threads in a warp are executed in parallel.

In CUDA the thread hierarchy is not just a Grid -> Block -> Thread but since some data-structure are represented in more than one dimension the hierarchy is represented in `dim3` which is a 3D vector (built in CUDA).

So for example:

```c
dim3 grid(2, 2, 1); // 2x2 grid
dim3 block(16, 16, 1); // 16x16 block

dim3 grid(2); // 2x1 grid
```

At first the usage of `dim3` might seem a bit confusing but it is very useful when you are working with 2D or 3D data structures. 

For examples in a matrix multiplication operation:

```c
dim3 grid(n/16, n/16, 1); // n x n matrix
dim3 block(16, 16, 1);  // 16 x 16 block

mat_mul<<<grid, block>>>(a, b, c, n);
```

In the code above we introduce the calling convention of a kernel function called `mat_mul` that is executed by a grid of blocks. Each block has 16x16 threads. The `n` is the size of the matrix.
The `<<<>>>` syntax is used to specify the grid and block size given in `dim3`.


## 2.3 Kernels

A kernel is a function that is executed on `device` (GPU). It is executed by many threads in parallel.

A kernel is defined in code by preceding the function with `__global__` keyword.

Here is an example of a simple kernel function:

```c
__global__ void my_kernel() {
    // kernel code
    printf("helo from kernel\n");
}
```

The kernel function is called by the host code. The host code is responsible for managing the GPU, allocating memory, copying data, etc.

### 2.3.1 Kernel Invocation

A kernel is invoked by the host code using the `<<<>>>` syntax.

Here is an example of invoking a kernel:

```c
int main() {
	my_kernel<<<1, 1>>>();
	return 0;
}
```

In the code above we invoke the kernel function `my_kernel` with a grid of 1 block and 1 thread.

### 2.3.2 Kernel Arguments

Kernels can take arguments just like regular functions.

Here is an example of a kernel function that takes arguments:

```c
__global__ void add(int a, int b, int * c) {
    *c = a + b;
}
```

The kernel function `add` takes two integer arguments `a` and `b` and a pointer to an integer `c`. The kernel function adds `a` and `b` and stores the result in `c`.

The kernel function is invoked by the host code like this:

```c
int main() {
  int c;
  int * dev_c;
  cudaMalloc((void**)&dev_c, sizeof(int));
  add<<<1, 1>>>(2, 7, dev_c);
  cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
  printf("2 + 7 = %d\n", c);
  cudaFree(dev_c);
  return 0;
}
```

In the code above we allocate memory on the device for the result of the addition operation. We then invoke the kernel function `add` with the arguments `2`, `7`, and a pointer to the device memory. We then copy the result back to the host memory and print it.

## 2.4 Thread Hierarchies / Indexing

Threads in CUDA are organized into blocks. Blocks are organized into a grid.

Threads are identified by a unique thread index. Blocks are identified by a unique block index.

Since the thread are organized in a 3D grid the thread index is represented by a `dim3` vector. and the block index is represented by a `dim3` vector. CUDA gives us access to variables that can be used to id the thread we are running.

The variables are:

1. `threadIdx`: which is a `dim3` vector that id the thread index (can be used for 1D, 2D, 3D indexing).
2. `blockIdx`: which is a `dim3` vector that id the block index (can be used for 1D, 2D, 3D indexing).
3. `blockDim`: which is a `dim3` vector that id the block dimensions (can be used for 1D, 2D, 3D indexing).
4. `gridDim`: which is a `dim3` vector that id the grid dimensions (can be used for 1D, 2D, 3D indexing).

Here is an example of how to use these variables:

```c
__global__ void mat_add(float A[N][N], float B[N][N], float C[N][N]) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) {
        C[i][j] = A[i][j] + B[i][j];
    }
}
```

In the code above we have a kernel function that adds two matrices `A` and `B` and stores the result in `C`. We use the `blockIdx` and `threadIdx` variables to id the thread index and block index.

## 2.5 Memory Hierarchy

CUDA threads can access multiple types of memory. We will go further into the details of each type of memory and the specifics in the next unit [CUDA Memory Model](/03_cuda_memory/README.MD). But here is an illustration of the memory hierarchy in CUDA:

![CUDA Memory Hierarchy](/images/06_image.png)

## 2.6 Heterogeneous Programming

CUDA is a heterogeneous programming model. It allows you to write code that can be executed on both the CPU and GPU. This means that we can hopefully try and leverage the best of both type of architecture.












