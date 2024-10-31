# CUDA Threads

In this unit, we will explore the CUDA Thread Model and how to use it to optimize our code. Understanding how to best utilize the threads (shapes and sizes) in different operations such as matrix multiplication, vector addition, etc. is crucial to writing efficient CUDA code.

## 4.1 Introduction

In CUDA, the idea is to try and parallelize the code as much as possible. The basic unit of execution in CUDA is the thread. Threads are grouped into blocks and blocks are grouped into grids. The number of threads in a block is limited by the hardware, but the number of blocks in a grid is not. Threads in a block can communicate with each other using shared memory, but threads in different blocks cannot. 

In this section we will try and demonstrate how to use thread with the different type of memory available in CUDA to optimize our code.

## 4.2 Vector Addition

We will write a basic vector addition kernel to show that basic usage of threads in CUDA. Here is the code:

```cpp
#include <iostream>
#include "cuda_runtime.h"

__global__ void vectorAdd(float * a, float * b, float * c, int n) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index < n) {
	    c[index] = a[index] + b[index];
	}
}
```

This is a very simple kernel that will receive two vectors `a` and `b` and add them together and store the result in vector `c`. 

Now we will see the part of the code that will call this kernel:

```cpp
void vector_add(float * a, float * b, float * c, int n) {
	float * d_a, * d_b, * d_c;
	
	// allocate memory on the device
	cudaMalloc(&d_a, n * sizeof(float));
	cudaMalloc(&d_b, n * sizeof(float));
	cudaMalloc(&d_c, n * sizeof(float));
	
	// copy data from host to device
	cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
	
	// define the grid and block size
	int block_size = 256;
	int grid_size = (n + block_size - 1) / block_size;
	
	// call the kernel
	vectorAdd<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
	
	// copy the result back to the host
	cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
	
	// free the memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

int main() {
	int n = 1000000;
	
	float * a = new float[n];
	float * b = new float[n];
	float * c = new float[n];
	
	create_random_vector(a, n);
	create_random_vector(b, n);
	
	vector_add(a, b, c, n);
	
	print_vector(a, 10);
	print_vector(b, 10);
	print_vector(c, 10);
	
	return 0;
}
```

You can see in the `add_vector` function that we allocate memory on the device, copy the data from the host to the device, call the kernel, copy the result back to the host and free the memory. This is the basic structure of most CUDA programs.

You can see more in the `04.01.vector_add` folder. In which we include:

1. `vectorAdd.cu`: The actual code with the kernel and utility functions.
2. `CMakeList.txt`: The `CMake` file to compile the code.
3. `run.sh`: A script to compile and run the code.

In order to run the code, go to the `04.01.vector_add` folder and run the following commands:

```bash
bash run.sh
```

See code [here](/04_cuda_threads/04.01_vectorAdd/vectorAdd.cu)

## 4.3 Matrix Multiplication

In this section we will see an optimization for the matrix multiplication algorithm. we will start by understanding the basic mathematical operation, simple CPU implementation, and then we will dive into the CUDA implementation using an algorithm called `tiling`.

### 4.3.1 Mathematical Operation

In linear algebra (a mathematical field that deals with vectors and matrices), the matrix multiplication is defined as follows:

given two matrices $A$ and $B$ of size $m \times n$ and $n \times p$ respectively, the matrix multiplication results in a new matrix $C$ of size $m \times p$ where:

$$C= \begin{bmatrix}  
c_{11} & c_{12} & \dots & c_{1p} \\  
c_{21} & c_{22} & \dots & c_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
c_{m1} & c_{m2} & \dots & c_{mp} \\
\end{bmatrix}$$

and each element $c_{ij}$ is calculated as follows:

$$c_{ij} = \sum_{k=1}^{n} a_{ik} \cdot b_{kj}$$


As you can see from the formula above, this operation is very good for parallelization. Each element in the result matrix $C$ can be calculated independently of the others. Which means that by using GPU and CUDA in particular we can achieve a significant speedup.

### 4.3.2 CPU Implementation

Here is a simple CPU implementation of the matrix multiplication algorithm:

```cpp
void matrix_multiply(float * a, float * b, float * c, int m, int n, int p) {
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < p; j++) {
      float sum = 0;
      for(int k = 0; k < n; k++) {
        sum += a[i * n + k] * b[k * p + j];
      }
      c[i * p + j] = sum;
    }
  }
}
```

This is a simple implementation of the matrix multiplication algorithm. It is not optimized and it is not parallelized. We will use this as a baseline to compare our CUDA implementation.

As you can see the basic CPU implementation is very slow in in CS term is running at $O(n^3)$. which means that the time it takes to run the algorithm is proportional to the cube of the size of the matrix. This is not good for large matrices.

### 4.3.3 CUDA Implementation

Now we will see the CUDA implementation of the matrix multiplication algorithm. We will use an optimization technique called `tiling` to achieve better performance.

> [!NOTE]
> The entire code for the matrix multiplication is in the `04.02.matrix_multiplication` folder, see [here](/04_cuda_threads/04.02_matMul/matMul.cu).

In order to understand the code we will first start be explaining the `tiling` technique.

#### Tiling

The idea behind tiling is to break the matrix into smaller submatrices and calculate the result for each submatrix. This is a very common optimization technique in matrix multiplication algorithms.

Here is a simple example to illustrate the idea:

Given two matrices $A$ and $B$ of size $4 \times 4$ and $4 \times 4$, we can break them into smaller submatrices of size $2 \times 2$ and calculate the result for each submatrix.

Here is an example:

$$A = \begin{bmatrix}  
a_{11} & a_{12} & a_{13} & a_{14} \\  
a_{21} & a_{22} & a_{23} & a_{24} \\
a_{31} & a_{32} & a_{33} & a_{34} \\
a_{41} & a_{42} & a_{43} & a_{44} \\
\end{bmatrix}
$$


$$B = \begin{bmatrix}  
b_{11} & b_{12} & b_{13} & b_{14} \\  
b_{21} & b_{22} & b_{23} & b_{24} \\
b_{31} & b_{32} & b_{33} & b_{34} \\
b_{41} & b_{42} & b_{43} & b_{44} \\
\end{bmatrix}
$$

Now we will break the matrices into smaller submatrices of size $2 \times 2$:

$$
A = \begin{bmatrix}  
a_{11} & a_{12} \\  
a_{21} & a_{22} \\
\end{bmatrix}
,
\begin{bmatrix}  
a_{13} & a_{14} \\  
a_{23} & a_{24} \\
\end{bmatrix}
,
\begin{bmatrix}
a_{31} & a_{32} \\  
a_{41} & a_{42} \\
\end{bmatrix}
,
\begin{bmatrix}  
a_{33} & a_{34} \\  
a_{43} & a_{44} \\
\end{bmatrix}
$$


$$
B = \begin{bmatrix}  
b_{11} & b_{12} \\  
b_{21} & b_{22} \\
\end{bmatrix}
,
\begin{bmatrix}  
b_{13} & b_{14} \\  
b_{23} & b_{24} \\
\end{bmatrix}
,
\begin{bmatrix}  
b_{31} & b_{32} \\  
b_{41} & b_{42} \\
\end{bmatrix}
,
\begin{bmatrix}
b_{33} & b_{34} \\  
b_{43} & b_{44} \\
\end{bmatrix}
$$

Now we will calculate the result for each submatrix:

$$C_{11} = A_{11} \cdot B_{11} + A_{12} \cdot B_{21}$$

$$C_{12} = A_{11} \cdot B_{12} + A_{12} \cdot B_{22}$$

$$C_{21} = A_{21} \cdot B_{11} + A_{22} \cdot B_{21}$$

etc...

This is the basic idea behind tiling. We will use this technique to optimize our CUDA implementation.

Here is a diagram of the idea:

![Tiling](/images/07_image.gif)

![Tiling](/images/08_image.gif)

### 4.3.4 CUDA Implementation

Now we will see the CUDA implementation of the matrix multiplication algorithm using the `tiling` technique.

Let's start by thinking about planning the algorithm:

1. We will break the matrices into smaller submatrices of size $T \times T$.
2. We will calculate the result for each submatrix.
3. We will use shared memory to store the submatrices.

- 1. Let's start by defining the kernel:

```cpp
template <int BLOCK_SIZE> __global__ void matMul(float * A, float * B, float * C, int wA, int wB);
```

This is the kernel that will calculate the result for each submatrix. The `BLOCK_SIZE` is the size of the submatrix. We will use this template parameter to define the size of the submatrix.

- 2. Let's continue by finding the all the needed indexes:

```cpp
// block index
int bx = blockIdx.x;
int by = blockIdx.y;

  
// thread index
int tx = threadIdx.x;
int ty = threadIdx.y;

  
// index of the first sub-matrix of A processed by the block
// example: if BLOCK_SIZE = 16, then wA = 256, then aBegin = 16 * 16 * 0 = 0
int aBegin = wA * BLOCK_SIZE * by;

  
// index of the last sub-matrix of A processed by the block
// example: if BLOCK_SIZE = 16, then wA = 256, then aEnd = 16 * 16 * 1 = 256
int aEnd = aBegin + wA - 1;

  
// step size used to iterate through the sub-matrices of A
// example: if BLOCK_SIZE = 16, then aStep = 16 * 16 = 256
int aStep = BLOCK_SIZE;

  
// index of the first sub-matrix of B processed by the block
// example: if BLOCK_SIZE = 16, then wB = 256, then bBegin = 16 * 16 * 0 = 0
int bBegin = BLOCK_SIZE * bx;
```

Let's see each variable:

- `bx` and `by`: The block index in the grid.
- `tx` and `ty`: The thread index in the block.
- `aBegin` and `aEnd`: The index of the first and last submatrix of `A` processed by the block.
- `aStep`: The step size used to iterate through the submatrices of `A`.
- `bBegin`: The index of the first submatrix of `B` processed by the block.

- 3. Let's use the `__shared__` memory and loop over the submatrices:

```cpp
// loop over all the sub-matrices of A and B required to compute the block sub-matrix
for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += BLOCK_SIZE) {
	// shared memory for the sub-matrix of A
	// remember: shared memory is shared among all the threads in the block and is bigger than local memory
	__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
	
	  
	// shared memory for the sub-matrix of B
	__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
	
	  
	// load the matrices from global memory to shared memory
	As[ty][tx] = A[a + wA * ty + tx];
	Bs[ty][tx] = B[b + wB * ty + tx];
	
	  
	// synchronize to make sure the matrices are loaded
	// the synchronization is needed because the threads are loading the data from global memory to shared memory
	// and if two threads are trying to access the same location in shared memory, then the data might be corrupted
	__syncthreads();
	
	  
	// perform the computation
	for (int k = 0; k < BLOCK_SIZE; k++) {
	Csub += As[ty][k] * Bs[k][tx];
	}
	
	  
	// synchronize to make sure that the preceding computation is done
	// the synchronization is needed because the threads are performing the computation
	// and if the threads are not synchronized, then the computation might be corrupted
	__syncthreads();

}
```

Let's break this down:

- We define the shared memory for the submatrices of `A` and `B`.
- We load the matrices from global memory to shared memory.
- We synchronize to make sure the matrices are loaded.
- We perform the computation.
- We synchronize to make sure the computation is done.

- 4. Let's calculate the indexes of the result matrix:

```cpp
// write the block sub-matrix to global memory
// each thread writes one element
int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
C[c + wB * ty + tx] = Csub;
```

This is the index of the result matrix.

Let's take a look at all the different types of memory used in this algorithm and try to understand the differences:

- `global memory`: This is the memory that is accessible by all the threads in the grid. It is the slowest type of memory.
- `shared memory`: This is the memory that is shared among all the threads in the block. It is faster than global memory.
- `local memory`: This is the memory that is private to each thread. It is the fastest type of memory.

The full code for the matrix multiplication algorithm is in the `04.02_matMul` folder, see [here](/04_cuda_threads/04.02_matMul/matMul.cu). there is also the `host` code to run the kernel.

## 4.4 Streams

In order to understand stream we need to understand the need of optimization. In a CUDA program we always perform the following steps:

1. Allocate memory on the device.
2. Copy the data from the host to the device.
3. Calling the kernel.
4. Copy the result back to the host.

So has we can see all program have in common is the copying the data from and to the device. So if we can optimize those operation we will improve the performance of our program.

The idea behind streams is to perform the copy operations in parallel with the kernel execution. This is done by creating multiple streams and using them to copy the data.

Here is a simple example to illustrate the idea:

```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

cudaMemcpyAsync(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice, stream1);
cudaMemcpyAsync(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice, stream2);

vectorAdd<<<grid_size, block_size, 0, stream1>>>(d_a, d_b, d_c, n);

cudaMemcpyAsync(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost, stream2);

cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

In a diagram we can show the idea as follows:

![Streams](/images/09_image.png)

