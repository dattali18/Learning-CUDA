# CUDA Threads

In this unit, we will explore the CUDA Thread Model and how to use it to optimize our code. Understanding how to best utilize the threads (shapes and sizes) in different operations such as matrix multiplication, vector addition, etc. is crucial to writing efficient CUDA code.

## 4.1 Introduction

In CUDA, the idea is to try and paralelllize the code as much as possible. The basic unit of execution in CUDA is the thread. Threads are grouped into blocks and blocks are grouped into grids. The number of threads in a block is limited by the hardware, but the number of blocks in a grid is not. Threads in a block can communicate with each other using shared memory, but threads in different blocks cannot. 

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

