# 0. Intro to GPU

In This unit, we will explore the microarchitecture of GPU and how it differs from CPU.

## Table of Content

- [CPU vs. GPU Architecture](#01-cpu-vs-gpu-architecture)
- [NVIDIA GPUs](#02-nvidia-gpus)
- [NVIDIA as a software company](#03-nvidia-as-a-software-company)
- [The usage of CUDA in the world of AI](#04-the-usage-of-cuda-in-the-world-of-ai)
## 0.1 CPU vs. GPU Architecture

CPU stands for Central Processing Unit, and it is a general purpose computing chip. A CPU generally has a few cores (in the biggest CPU in the world currently has 64 cores). Each core can execute a single instruction at a time. CPU cores are optimized for serial processing, which means they are good at executing a single instruction at a time. 

![GPU vs CPU](/images/01_image.png)

In the above image we can try and see the difference between the design choices of a CPU vs. GPU, in the CPU it is optimized for serial processing, while the GPU is optimized for parallel processing. 

We can also distinguished the difference between the two chips by looking at the caches and how they are used. In the CPU each core as it's own L1 caches and L2 caches, and all cores share the L3 caches. In the GPU a clusters of cores (in NVIDIA called a SM - Streaming Multiprocessor) share the L1 and L2 caches, and the L3 cache is shared across all SMs (some GPU don't have an L3 cache at all).

There is also a difference in the size and speed of the different caches.


|     | CPU                                                                                                                | GPU                                                                                   |
| --- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| L1  | Typically smaller (e.g., 32KB-64KB per core) / Very fast (directly accessed by core)                               | Often larger (e.g., 32KB-64KB per SM) / Very fast (directly accessed by SM)           |
| L2  | Larger than L1 (e.g., 256KB-512KB per core) / Slower than L1, but faster than main memory                          | Larger than L1 (e.g., 512KB-1MB per SM) / Slower than L1, but faster than main memory |
| L3  | Largest cache level (e.g., 8MB-32MB shared by multiple cores) / Slower than L1 and L2, but faster than main memory | Smaller or nonexistent                                                                |


### 0.1.1 Key Differences in Caches:

- **Cache Hierarchy:** CPUs often have a more complex cache hierarchy with L1, L2, and L3 levels, while GPUs typically have only L1 and L2 caches.
- **Cache Size:** GPUs tend to have larger L1 and L2 caches per processing unit (SM) compared to CPUs per core. This is because GPUs are designed for parallel processing and need to store more data for concurrent operations.
- **Cache Sharing:** CPUs may share L3 cache among multiple cores, while GPUs often have separate L1 and L2 caches for each SM.

**Why GPUs have larger caches:**

- **Parallel Processing:** GPUs handle many tasks simultaneously, requiring more data to be readily available.
- **Memory Access Latency:** GPUs often have higher memory access latency compared to CPUs, so larger caches can help reduce the frequency of memory accesses.

### 0.1.2 The Usage of GPU

In this section will will take a look at how GPUs are used mainly through the lens of AI & DL.

**AI & Deep Learning:**

- **Training Neural Networks:** GPUs are widely used to accelerate the training of deep learning models. The parallel processing capabilities of GPUs enable faster matrix operations and optimization algorithms, leading to quicker training times.
- **Inference:** GPUs are also used for deep learning inference, where pre-trained models make predictions on new data. GPUs can efficiently process large volumes of data and perform complex computations required for inference tasks.
- **Model Optimization:** GPUs are essential for optimizing deep learning models through techniques like hyperparameter tuning, architecture search, and model pruning. The computational power of GPUs enables researchers to experiment with different model configurations and training strategies.
- **AI Research:** GPU-accelerated computing has revolutionized AI research by enabling scientists to explore complex neural network architectures, train models on large datasets, and develop innovative AI applications.

Why? Because all of the operation above are based on  **linear algebra** which can be parallelized. This is why GPUs are so good at AI & DL.

### 0.1.3 CPU GPU communication

In the world of AI & DL, the CPU is used to manage the data and the GPU is used to do the computation. The CPU will load the data into the GPU, and the GPU will do the computation. The CPU will then take the result and do some post-processing on the result. This is the general workflow of AI & DL.

The work-flow looks like so:

1. CPU prepare and load the data from disk.
2. CPU call a low level API to transfer the data from the RAM (host) to to the GPU (device).
3. CPU call a kernel (operation on GPU) to do the computation.
4. GPU do the computation.
5. CPU call a low level API to transfer the result from the GPU to the CPU.
6. CPU do some post-processing on the result.
7. CPU save the result to disk.

## 0.2 NVIDIA GPUs

NVIDIA is a chip maker that produce GPUs, we will take a look a their product over the years to understand the evolution of the GPU as a product.

### Kepler Architecture (2012)

- **Key Features:**
    - CUDA Cores: 1,536-3,072
    - Streaming Multiprocessors (SMs): 15-30
    - Memory Bus Width: 256-bit
    - Architecture: Fermi-based
- **Notable GPUs:** GeForce GTX 600 series, Quadro K series

### Maxwell Architecture (2014)

- **Key Features:**
    - CUDA Cores: 1,280-2,560
    - Streaming Multiprocessors (SMs): 12-24
    - Memory Bus Width: 128-bit
    - Architecture: New microarchitecture
- **Notable GPUs:** GeForce GTX 700 series, Quadro M series

### Pascal Architecture (2016)

- **Key Features:**
    - CUDA Cores: 1,280-3,840
    - Streaming Multiprocessors (SMs): 12-30
    - Memory Bus Width: 128-bit
    - Architecture: New microarchitecture
- **Notable GPUs:** GeForce GTX 1000 series, Quadro P series

### Volta Architecture (2017)

- **Key Features:**
    - CUDA Cores: 3,584-6,144
    - Streaming Multiprocessors (SMs): 21-40
    - Memory Bus Width: 256-bit
    - Architecture: New microarchitecture
- **Notable GPUs:** Tesla V100, Quadro GV100

### Turing Architecture (2018)

- **Key Features:**
    - CUDA Cores: 2,048-4,608
    - Streaming Multiprocessors (SMs): 12-24
    - Memory Bus Width: 192-bit
    - Architecture: New microarchitecture with ray tracing and deep learning capabilities
- **Notable GPUs:** GeForce RTX 2000 series, Quadro RTX series

### Ampere Architecture (2020)

- **Key Features:**
    - CUDA Cores: 3,072-8,192
    - Streaming Multiprocessors (SMs): 18-40
    - Memory Bus Width: 256-bit
    - Architecture: New microarchitecture with enhanced ray tracing and deep learning capabilities
- **Notable GPUs:** GeForce RTX 3000 series, Quadro RTX A6000

### Ada Lovelace Architecture (2022)

- **Key Features:**
    - CUDA Cores: 4,096-18,432
    - Streaming Multiprocessors (SMs): 24-80
    - Memory Bus Width: 256-bit
    - Architecture: New microarchitecture with advanced ray tracing, AI, and graphics capabilities
- **Notable GPUs:** GeForce RTX 4000 series, Quadro RTX A5000

### Hopper Architecture (2022)

- **Key Features:**
    - CUDA Cores: 4,352-10,752
    - Streaming Multiprocessors (SMs): 26-40
    - Memory Bus Width: 256-bit
    - Architecture: New microarchitecture designed for data center and high-performance computing applications
- **Notable GPUs:** H100

> [!NOTE]
> This table provides a general overview of key features and does not include all variations within each architecture. Specific GPU models may have additional features or performance optimizations.


There is a new architecture coming out called **Blackwell** which is a new microarchitecture designed for edge computing and IoT applications. It is expected to have lower power consumption and a smaller form factor compared to previous architectures.

## 0.3 NVIDIA as a software company

NVIDIA is not a hardware company anymore, they are also a software company. They have developed a lot of software to help developers and researchers to develop software for their GPUs. Some of the software that NVIDIA has developed are:

- **CUDA:** A parallel computing platform and programming model that enables developers to use NVIDIA GPUs for general-purpose processing.
- **cuDNN:** The NVIDIA CUDA Deep Neural Network library provides high-performance primitives for deep learning frameworks.
- **TensorRT:** An SDK for high-performance deep learning inference that includes a deep learning inference optimizer and runtime.
- **NVIDIA Nsight Systems:** A system-wide performance analysis tool designed to visualize application performance and optimize software for NVIDIA GPUs.
- **NVIDIA Nsight Compute:** A CUDA kernel profiler that provides detailed performance metrics and API debugging capabilities.
- **NVIDIA Nsight Graphics:** A graphics debugging and profiling tool for applications running on NVIDIA GPUs.
- **NVIDIA DeepStream SDK:** A toolkit for building scalable AI-powered video analytics applications.
- **NVIDIA Triton Inference Server:** An open-source inference serving software that simplifies the deployment of AI models at scale.
- **NVIDIA Jarvis:** An application framework for building conversational AI services.

This gave NVIDIA the edge over other company.

**How CUDA Gave NVIDIA an Advantage:**

1. **Accessibility:** CUDA provided a relatively easy-to-use programming model for developers to harness the power of NVIDIA GPUs. By leveraging familiar C-like syntax and extensions, developers could quickly adapt their existing code or create new applications tailored to GPU acceleration.
2. **Ecosystem Development:** The CUDA platform fostered a vibrant ecosystem of software tools, libraries, and frameworks. This ecosystem provided developers with a wide range of resources to accelerate their workflows, from deep learning frameworks like TensorFlow and PyTorch to scientific computing libraries like cuBLAS and cuFFT.
3. **Performance Leadership:** CUDA-enabled applications consistently demonstrated superior performance compared to CPU-only implementations. The massively parallel architecture of NVIDIA GPUs, combined with optimized CUDA libraries, allowed for significant speedups in computationally intensive tasks.
4. **Intellectual Property:** CUDA became a cornerstone of NVIDIA's intellectual property portfolio. The company's control over the CUDA platform gave it a competitive edge, ensuring that its GPUs were the preferred choice for developers seeking to leverage GPU acceleration.
5. **Vertical Market Penetration:** CUDA's versatility enabled NVIDIA to expand its reach beyond traditional gaming and graphics markets. The platform found applications in fields such as healthcare, finance, and automotive, solidifying NVIDIA's position as a leading provider of high-performance computing solutions.

This give us a better understanding of the CUDA as a project as a whole. We now understand what it is and why it is important.

## 0.4 The usage of CUDA in the world of AI

Here is a short list of AI and ML frameworks that use CUDA for accelerating and optimizing their operations:

1. **TensorFlow** - An open source library for numerical operation with API for DNN and ML. Developed at Google.
2. **PyTorch** - An open source library for numerical operation with API for DNN and ML. Developed at Facebook.
3. **Caffe** - A deep learning framework developed by the Berkeley Vision and Learning Center.
4. **MXNet** - A deep learning framework developed by Apache.
5. **Keras** - A high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.
6. **JAX** - A research-focused neural network library developed by Google.
7. **MLX** - A deep learning framework developed by Apple for Apple Silicon.

In this course we will mainly focus on the the low-level usage of CUDA mainly focus towards the CUDA C++ API in relation to AI & DL (not for graphics or other usage of GPUs).


# 1. Intro to CUDA

In this unit, we will explore CUDA what is it and what problem it solves.

## Table of Contents

- [What is CUDA](#11-what-is-cuda)
- [Introduction to CUDA](#12-introduction-to-cuda)
- [CUDA Programming Model](#13-cuda-programming-model)
- [Key CUDA Concepts](#14-key-cuda-concepts)
- [SM](#15-sm)
- [Microarchitecture](#16-microarchitecture)
- [CUDA Core](#17-cuda-core)
- [CUDA Memory Model](#18-cuda-memory-model)
- [Thread Wrap](#19-thread-wrap)
- [SIMD](#110-simd)

## 1.1 What is CUDA?

**CUDA** (Compute Unified Device Architecture) is a parallel computing platform and application programming interface (API) model created by Nvidia. It allows software developers to use a CUDA-enabled graphics processing unit (GPU) for general-purpose processing.

Context: Before CUDA there was (and still is) other way for programmer to access the GPU such as: OpenGL, OpenCL, DirectCompute, etc. But however CUDA solves a problem that they all had in common: they wan't design for the specific GPU architecture. But in the case of CUDA it was designed and built from the ground up to be used with Nvidia's GPU architecture, those making it faster and more efficient.

## 1.2 Introduction to CUDA

CUDA is a parallel computing platform and API model, it differ from normal CPU programming in a few key ways:

1. **Massive Parallelism**: While a CPU has a few cores (2-8 cores), a GPU has thousands of cores. This allows for massive parallelism, which is the ability to run many threads at the same time.

2. **SIMD**: GPUs are designed to run the same instruction on multiple data (SIMD). This is different from CPU which runs different instructions on different data.

3. **Memory Hierarchy**: GPUs have a different memory hierarchy than CPU. They have a global memory, shared memory, constant memory, texture memory, etc.

4. **Thread Hierarchy**: GPUs have a different thread hierarchy than CPU. They have blocks, warps, threads, etc.

Let's give some examples:

```cpp
// CPU code
void vector_add(int * a , int * b, int * c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// GPU code 
__global__ void vector_add(int * a , int * b, int * c, int n) {
    int i = threadIdx.x;
    if( (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

As you can see from the very simple example above of a vector addition operation, the programming model is very different. In the CPU code, we have a for loop that iterates over the entire array and adds the elements. In the GPU code, we have a kernel that is executed by thousands of threads. Each thread is responsible for adding one element of the array.

## 1.3 CUDA Programming Model

The CUDA programming model is based on the following concepts:

1. **Host**: The CPU is called the host. The host is responsible for managing the GPU, allocating memory, copying data, etc.

2. **Device**: The GPU is called the device. The device is responsible for executing the kernel code.

3. **Kernel**: A kernel is a function that is executed on the device. It is executed by many threads in parallel.

4. **Thread**: A thread is a single execution unit. Threads are organized into blocks.

5. **Block**: A block is a group of threads. Blocks are organized into a grid.

6. **Grid**: A grid is a group of blocks.


Here is an illustration of the CUDA programming model:

![CUDA Programming Model](/images/02_image.png)

In the illustration above you can see a few things:

1. The **host** code is executed in a line by line fashion by the CPU.

2. The **device** has a hierarchy of thread separated into Grid -> Blocks -> Threads.

3. The memory in the **device** is separated into very distinct types and each has it own purpose rule and scope.

4. The **kernel** is a function that is executed by many threads in parallel. It is called by the host code.

## 1.4 Key CUDA Concepts

1. **Device Memory**: Memory on the device (GPU). It is separate from the host memory (CPU).

2. **Host Memory**: Memory on the host (CPU). It is separate from the device memory (GPU).

3. **Kernel**: A function that is executed on the device (GPU).

4. **Thread**: A single execution unit. Threads are organized into blocks.

5. **Block**: A group of threads. Blocks are organized into a grid.

6. **Grid**: A group of blocks.

7. **Shared Memory**: Memory that is shared between threads in a block.

8. **Constant Memory**: Memory that is read-only and cached.

9. **Texture Memory**: Memory that is optimized for 2D spatial locality. (not used in AI & DL). (Not used in AI)

10. **Unified Memory**: Memory that is shared between the host and device.

11. **Streams**: A sequence of commands that are executed in order.

12. **Events**: A way to measure time on the device.

13. **SM** (Streaming Multiprocessor): A group of CUDA cores.

14. **Warp**: A group of 32 threads that are executed in parallel.

15. **SIMD**: Single Instruction, Multiple Data. A way to run the same instruction on multiple data.

16. **SIMT**: Single Instruction, Multiple Threads. A way to run the same instruction on multiple threads.


## 1.5 SM

SM (Streaming Multiprocessor) is a group of CUDA cores. It is the basic building block of the GPU architecture. Each SM contains multiple CUDA cores that can execute instructions in parallel. The number of CUDA cores per SM can vary depending on the GPU model and architecture. SMs are responsible for executing the kernel code and managing the execution of threads.

The SM contains the following components:

1. **L-Cache**: A small, fast memory cache that stores frequently accessed data and instructions.
2. **MT Issue**: The instruction issue unit that schedules and dispatches instructions to the CUDA cores.
3. **C-Cache**: A shared memory cache that stores data shared among CUDA cores in the SM.
4. **SPs**: CUDA cores that perform arithmetic and logical operations on data.
5. **SFUs**: Special Function Units that perform complex operations like transcendental functions.
6. **DP**: Double-precision units that perform double-precision floating-point operations.
7. **Shared Memory**: A fast, on-chip memory that is shared among threads in the same block.


![GPU Architecture](/images/14_image.png)

## 1.6 Microarchitecture

![Microarchitecture](/images/15_image.png)

In the diagram we can see the whole picture of a GPU. We have the following component:

- **CPU**: The central processing unit (CPU) is the main processor in a computer. It executes instructions and performs calculations.
- **Host Memory (RAM)**: The host memory is the main memory of the computer. It stores data and instructions that are used by the CPU.
- **PCIe Bus**: The PCIe bus is a high-speed serial bus that connects the CPU to the GPU.
- **GPU**:  The graphics processing unit (GPU) is a specialized processor that is designed to render graphics and perform parallel computations.
	- **Global Memory (VRAM)**: The global memory is the main memory of the GPU. It stores data and instructions that are used by the GPU.
	- **GPU Memory Bus**: The GPU memory bus is a high-speed bus that connects the GPU to the global memory.
	- **L2 Cache**: The L2 cache is a small, fast memory cache that stores frequently accessed data and instructions.
	- **SM**: The streaming multiprocessor (SM) is a group of CUDA cores that can execute instructions in parallel.
		- **Shared Memory**: The shared memory is a fast, on-chip memory that is shared among threads in the same block.
		- **L1 Cache**: The L1 cache is a small, fast memory cache that stores frequently accessed data and instructions.
		- **Registers**: : Registers are small, fast memory locations that store data and instructions that are used by the CUDA cores.
		- **CUDA Cores**: CUDA cores are small, specialized processors that perform arithmetic and logical operations on data.

## 1.7 CUDA Core

A CUDA core is a small, specialized processor that is designed to perform arithmetic and logical operations on data. CUDA cores are the basic building blocks of the GPU architecture. Each CUDA core can execute a single instruction in parallel with other CUDA cores. The number of CUDA cores in a GPU can vary depending on the GPU model and architecture.

What is interesting about CUDA cores is that they are designed to be highly parallel. This means that they can execute many instructions in parallel, which allows them to perform complex calculations quickly and efficiently. CUDA cores are optimized for parallel processing and are designed to handle large amounts of data in parallel.

![CUDA Core](/images/16_image.webp)

As we can see inside the CUDA core there is:

- **Dispatch Port**: The dispatch port is responsible for scheduling and dispatching instructions to the CUDA core.
- **Operand Collector**: The operand collector is responsible for fetching the operands required for the instruction.
- **FP Unit**: The floating-point unit (FP unit) is responsible for performing arithmetic operations on floating-point numbers.
- **INT Unit**: The integer unit (INT unit) is responsible for performing arithmetic operations on integer numbers.
- **Result Collector**: The result collector is responsible for collecting the result of the instruction and storing it in the register file.

## 1.8 CUDA Memory Model

Take a look at the dedicated unit for it [here](/03_cuda_memory/README.md).

## 1.9 Thread Wrap

A warp is a group of 32 threads that are executed in parallel on the GPU. The warp is the basic unit of execution on the GPU. Each thread in the warp executes the same instruction at the same time. This allows the GPU to perform operations on multiple data points simultaneously.

This means that sometime we require the number of threads to be a multiple of 32. This is because the GPU executes threads in warps of 32 threads. If the number of threads is not a multiple of 32, the GPU will create a warp with the remaining threads and execute them in parallel.

## 1.10 SIMD

There is a dedicated unit explaining the concept of SIMD [here](/general_knowledge/simd.md).

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












# 3. CUDA Memory Model

In this unit, we will explore the CUDA Memory Model and it's hierarchy. we will try and understand the different types of memory available in CUDA and how to use them best to optimize our code.
## Table of Contents

- [Introduction](#31-introduction)
- [Memory Hierarchy](#32-memory-hierarchy)

## 3.1 Introduction

In the CUDA programming model there are a few type of accessible memory and each has it's pros and cons. scope and rules. Here are the main types of memory in CUDA:

1. **Global Memory**: Memory that is accessible by all threads in all blocks. It is the slowest memory in terms of access speed. Usually used for large data structures.

2. **Shared Memory**: Memory that is shared by all threads in a **block**. It is the fastest memory in terms of access speed. Usually used for temporary data.

3. **Local Memory**: Memory that is private to each thread. It is the slowest memory in terms of access speed. Usually used for local variables.

4. **Constant Memory**: Memory that is read-only and is accessible by all threads in all blocks. It is cached and has a very high access speed. Usually used for constants.

5. **Local Memory**: Memory that is private to each thread. It is the slowest memory in terms of access speed. Usually used for local variables.

6. **Texture Memory**: Memory that is optimized for 2D spatial locality. It is cached and has a high access speed. Usually used for image processing. (not usually used in DL).

7. **Registers**: Memory that is private to each thread. It is the fastest memory in terms of access speed. Usually used for local variables.

## 3.2 Memory Hierarchy

Here is a diagram of the memory hierarchy in CUDA:

![CUDA Memory Hierarchy](/images/03_image.png)

The same can be represented in a table:


|           | On/Off Chip | Access | Scope  | Liftime         |
| --------- | ----------- | ------ | ------ | --------------- |
| Registers | on          | R/W    | Thread | Thread          |
| Shared    | on          | R//W   | Block  | Block           |
| Local     | off         | R//W   | Thread | Thread          |
| Global    | off         | R//W   | Global | Host Controlled |
| Constant  | off         | R      | Gloabl | Host Controlled |

# CUDA Threads

In this unit, we will explore the CUDA Thread Model and how to use it to optimize our code. Understanding how to best utilize the threads (shapes and sizes) in different operations such as matrix multiplication, vector addition, etc. is crucial to writing efficient CUDA code.

## Table of Contents

- [Introduction](#41-introduction)
- [Vector Addition](#42-vector-addition)
- [Matrix Multiplication](#43-matrix-multiplication)
- [Streams](#44-streams)

### Code Example

- [Vector Addition](/04_cuda_threads/04.01_vectorAdd)
- [Matrix Multiplication](/04_cuda_threads/04.02_matMul)

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

You can see more in the `04.01_vectorAdd` folder. In which we include:

1. `vectorAdd.cu`: The actual code with the kernel and utility functions.
2. `CMakeList.txt`: The `CMake` file to compile the code.
3. `run.sh`: A script to compile and run the code.

In order to run the code, go to the `04.01_vectorAdd` folder and run the following commands:

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
> The entire code for the matrix multiplication is in the `04.02_matMul` folder, see [here](/04_cuda_threads/04.02_matMul/matMul.cu).

In order to understand the code we will first start be explaining the `tiling` technique.

#### Tiling

The idea behind tiling is to break the matrix into smaller sub-matrices and calculate the result for each sub-matrix. This is a very common optimization technique in matrix multiplication algorithms.

Here is a simple example to illustrate the idea:

Given two matrices $A$ and $B$ of size $4 \times 4$ and $4 \times 4$, we can break them into smaller sub-matrices of size $2 \times 2$ and calculate the result for each sub-matrix.

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

Now we will break the matrices into smaller sub-matrices of size $2 \times 2$:

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

# 5.0 CUDA Synchronization

In this unit, we will explore the CUDA API for thread synchronization. We will learn how to use the different synchronization primitives provided by CUDA to write efficient and correct parallel code. We will also touch on the async operations in CUDA and the usage of atomic operations.

In CUDA there is two ways of doing things:

- **Synchronous**: The CPU waits for the GPU to finish the task.
- **Asynchronous**: The CPU does not wait for the GPU to finish the task.

## Table of Contents

- [Synchronous Operations](#51-synchronous-operations)
- [Asynchronous Operations](#52-asynchronous-operations)
- [Atomic Operations](#53-atomic-operations)
- [Thread Synchronization](#54-thread-synchronization)
- [Conclusion](#55-conclusion)

## 5.1 Synchronous Operations

In synchronous operations, the CPU waits for the GPU to finish the task. This is the default behavior of CUDA. The CPU will wait for the GPU to finish the task before moving on to the next task. This is useful when you want to synchronize the CPU and GPU.

```cpp
#include <iostream>

__global__ void kernel() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    kernel<<<1, 10>>>();
    cudaDeviceSynchronize();
    std::cout << "Hello from CPU" << std::endl;
    return 0;
}
```

In the above code, the kernel is launched with 1 block and 10 threads. The CPU will wait for the GPU to finish the kernel execution before printing "Hello from CPU".

## 5.2 Asynchronous Operations

In asynchronous operations, the CPU does not wait for the GPU to finish the task. This is useful when you want to overlap the CPU and GPU tasks.

```cpp
#include <iostream>

__global__ void kernel() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    kernel<<<1, 10>>>();
    std::cout << "Hello from CPU" << std::endl;
    cudaDeviceSynchronize();
    return 0;
}
```

In the above code, the kernel is launched with 1 block and 10 threads. The CPU will not wait for the GPU to finish the kernel execution and will print "Hello from CPU" before the kernel execution is complete.

## 5.3 Atomic Operations

Atomic operations are used to perform operations on shared memory locations. Atomic operations are used

```cpp
#include <iostream>

__global__ void kernel(int *a) {
    atomicAdd(a, 1);
}

int main() {
    int *d_a;
    cudaMalloc(&d_a, sizeof(int));
    cudaMemset(d_a, 0, sizeof(int));
    kernel<<<1, 10>>>(d_a);
    cudaDeviceSynchronize();
    int a;
    cudaMemcpy(&a, d_a, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "a: " << a << std::endl;
    return 0;
}
```

In the above code, the kernel is launched with 1 block and 10 threads. The kernel increments the value of `a` by 1 using the atomicAdd function. The atomicAdd function is used to perform atomic addition on the shared memory location `a`.

## 5.4 Thread Synchronization

As we saw in the Matrix Multiplication example, we can use thread synchronization to synchronize the threads in a block. We can use the `__syncthreads()` function to synchronize the threads in a block.

This is sometime useful to do when a wrap of thread in a block are working on the same data chunk and we don't want to have data inconsistency.

For example:

```cpp
#include <iostream>

__global__ void kernel(int *a) {
    int tid = threadIdx.x;
    a[tid] = tid;
    __syncthreads();
    a[tid] = a[tid] + 1;
}


int main() {
    int *d_a;
    cudaMalloc(&d_a, 10 * sizeof(int));
    kernel<<<1, 10>>>(d_a);
    cudaDeviceSynchronize();
    int a[10];
    cudaMemcpy(a, d_a, 10 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i++) {
        std::cout << "a[" << i << "]: " << a[i] << std::endl;
    }
    return 0;
}
```

In the above code, the kernel is launched with 1 block and 10 threads. The kernel initializes the shared memory location `a` with the thread id. The `__syncthreads()` function is used to synchronize the threads in the block. After synchronization, the threads increment the value of `a` by 1.


But has we saw there is also a place for atomic operation sometimes. For example, if we want to increment a value in a shared memory location, we can use atomic operations.

```cpp
#include <iostream>

__global__ void kernel(int *a) {
    int tid = threadIdx.x;
    atomicAdd(a, tid);
}
```

In the above code, the kernel is launched with 1 block and 10 threads. The kernel increments the value of `a` by the thread id using the atomicAdd function.


## 5.5 Conclusion

In this unit, we explored the CUDA API for thread synchronization. We learned how to use the different synchronization primitives provided by CUDA to write efficient and correct parallel code. We also touched on the async operations in CUDA and the usage of atomic operations.
# 6.0 CUDA Tools

In this units, we will explore tools provided by NVIDIA for profiling and optimization such as Nsight Compute (for profiling kernels).

Since we are not able to run Nsight Compute on the cloud, we will provide you with a brief overview of the tool and how to use it on your local machine.

## Table of Contents

- [Nsight Compute](#61-nsight-compute)
- [Nsight System](#62-nsight-system)

## 6.1 Nsight Compute

Nsight Compute is a profiler for CUDA applications. It allows you to analyze the performance of your CUDA kernels and identify bottlenecks in your code.

### Compiling the Code

To use Nsight Compute, you need to compile your CUDA code with the `-lineinfo` flag to include line number information in the executable. This will allow Nsight Compute to map the performance data to the source code.

```bash
nvcc -lineinfo my_kernel.cu -o my_kernel
```

### Profiling Kernels

To profile your CUDA kernels using Nsight Compute, you can use the following command:

```bash
nsys profile --stats=true --force-overwrite true -o my_kernel_report ./my_kernel
```

This command will generate a report file `my_kernel_report.qdrep` that contains the profiling data for your CUDA kernels.


### Analyzing the Report

You can open the report file `my_kernel_report.qdrep` using Nsight Compute to analyze the performance data. Nsight Compute provides a graphical interface to visualize the performance metrics of your CUDA kernels and identify bottlenecks in your code.

We will put images of the interface here. But due to our limited resources we are not able to show the details of how to use and understand the tool fully.

![Nsight Compute](/images/10_image.jpg)

Example for wrap occupancy:

![Nsight Compute](/images/11_image.webp)


### Conclusion

In this unit, we explored the Nsight Compute profiler for CUDA applications. We learned how to compile our CUDA code with line number information and profile our kernels using Nsight Compute. We also learned how to analyze the profiling data to identify performance bottlenecks in our code.

There many more to explore for this tool that is very useful for optimization of you CUDA code.

## 6.2 Nsight System

We will not dive into the Nsight System in this unit, but it is also a very useful tool to profile the whole system and not only the CUDA kernels.
# 7.0 CUDA Libraries

In this unit, we will touch on the different library provided by NVIDIA to help us write better and faster CUDA code such as cuBLAS, cuDNN, and more. We will also explore how to use these libraries in our code.

CUDA Libraries such as:

- cuBLAS - CUDA Basic Linear Algebra Subprograms
- cuFFT - CUDA Fast Fourier Transform
- cuRAND - CUDA Random Number Generation
- cuSPARSE - CUDA Sparse Matrix
- cuDNN - CUDA Deep Neural Network
- cuTENSOR - CUDA Tensor Library
- cuSOLVER - CUDA Solver Library
- cuBLASLt - cuBLAS Light
- and many more

For the time being we will take a look at cuBLAS and cuFFT.

## Table of Contents

- [cuBLAS](#71-cublas)
- [cuFFT](#72-cufft)
- [Conclusion](#73-conclusion)

## 7.1 cuBLAS

cuBLAS is a library provided by NVIDIA for performing basic linear algebra operations on NVIDIA GPUs. It provides a set of routines that are optimized for NVIDIA GPUs to perform matrix operations such as matrix multiplication, matrix addition, and more.

### Links

-  The official documentation from NVIDIA [here](https://docs.nvidia.com/cuda/cublas/index.html).
-  Code example from NVIDIA [here](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/4_CUDA_Libraries).
- Code example from the CUDA course [here](https://github.com/Infatoshi/cuda-course/tree/master/06_CUDA_APIs/01%20CUBLAS)

The basic operation that could be found in the library are:

- Matrix-vector multiplication
- Matrix-matrix multiplication
- Matrix addition
- Matrix subtraction
- Matrix scaling
- Matrix transpose
- Matrix-vector addition
- Matrix-vector subtraction
- Matrix-vector scaling
- Matrix-vector multiplication
- ...

### Example

```cpp
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    const int N = 3;
    float *h_A, *h_x, *h_y;
    float *d_A, *d_x, *d_y;

    h_A = (float *)malloc(N * N * sizeof(float));
    h_x = (float *)malloc(N * sizeof(float));
    h_y = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N * N; i++) {
        h_A[i] = i;
    }

    for (int i = 0; i < N; i++) {
        h_x[i] = i;
    }

    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemv(handle, CUBLAS_OP_N, N, N, &alpha, d_A, N, d_x, 1, &beta, d_y, 1);

    cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        std::cout << "y[" << i << "]: " << h_y[i] << std::endl;
    }

    free(h_A);
    free(h_x);
    free(h_y);
    
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);

    cublasDestroy(handle);

    return 0;
}
```

Explanation:

- We first create a handle for cuBLAS using `cublasCreate(&handle)`.
- We allocate memory for the matrices `A`, `x`, and `y` on the host.
- We initialize the matrices `A` and `x` with some values.
- We allocate memory for the matrices `A`, `x`, and `y` on the device.
- We copy the matrices `A` and `x` from the host to the device.
- We define the scalar values `alpha` and `beta`.
- We perform the matrix-vector multiplication using `cublasSgemv`.
- We copy the result vector `y` from the device to the host.
- We print the result vector `y`.
- We free the memory allocated on the host and device.
- We destroy the cuBLAS handle.

In math writing the operation is:

$$Ax = \alpha \cdot y + \beta$$

where:

- $A$ is a matrix of size $N \times N$
- $x$ is a vector of size $N$
- $y$ is a vector of size $N$
- $\alpha$ is a scalar value
- $\beta$ is a scalar value

All operations using the cuBLAS library has the same structure as the above example.

1. Create a handle
2. Allocate memory for the matrices on the host
3. Initialize the matrices with some values
4. Allocate memory for the matrices on the device
5. Copy the matrices from the host to the device
6. Define the scalar values
7. Perform the operation using the cuBLAS function
8. Copy the result from the device to the host
9. Free the memory allocated on the host and device
10. Destroy the cuBLAS handle

> An `handle` is a pointer to the cuBLAS library that is used to perform operations on the GPU.

> All operation in the cuBLAS library are heterogeneous, meaning that the operation is done on the GPU. And they can be called from the host code.


## 7.2 cuFFT

cuFFT is a library provided by NVIDIA for performing fast Fourier transform (FFT) operations on NVIDIA GPUs. It provides a set of routines that are optimized for NVIDIA GPUs to perform FFT operations on complex and real data.

> FFT or Fast Fourier Transform is an algorithm that computes the discrete Fourier transform (DFT) of a sequence, or its inverse (IDFT). Fourier analysis converts a signal from its original domain (often time or space) to a representation in the frequency domain and vice versa.


FFT formula in math is:

$$X_k = \sum_{n=0}^{N-1} x_n \cdot e^{-2\pi i \frac{kn}{N}}$$

where:

- $X_k$ is the $k^{th}$ frequency component
- $x_n$ is the $n^{th}$ input signal
- $N$ is the number of samples
- $i$ is the imaginary unit

### Links

-  The official documentation from NVIDIA [here](https://docs.nvidia.com/cuda/cufft/index.html).
- Code example from NVIDIA [here](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/4_CUDA_Libraries/simpleCUFFT)

### Example

```cpp
#include <iostream>
#include <cuda_runtime.h>
#include <cufft.h>

int main() {
    cufftHandle plan;
    cufftComplex *h_data, *d_data;
    int N = 8;

    h_data = (cufftComplex *)malloc(N * sizeof(cufftComplex));
    cudaMalloc(&d_data, N * sizeof(cufftComplex));

    for (int i = 0; i < N; i++) {
        h_data[i].x = i;
        h_data[i].y = 0;
    }

    cudaMemcpy(d_data, h_data, N * sizeof(cufftComplex), cudaMemcpyHostToDevice);

    cufftPlan1d(&plan, N, CUFFT_C2C, 1);

    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);

    cudaMemcpy(h_data, d_data, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        std::cout << "data[" << i << "]: " << h_data[i].x << " + " << h_data[i].y << "i" << std::endl;
    }

    free(h_data);
    cudaFree(d_data);

    cufftDestroy(plan);

    return 0;
}
```


Explanation:

- We first create a plan for cuFFT using `cufftPlan1d`.
- We allocate memory for the complex data on the host and device.
- We initialize the complex data with some values.
- We allocate memory for the complex data on the device.
- We copy the complex data from the host to the device.
- We execute the forward FFT using `cufftExecC2C`.
- We copy the result complex data from the device to the host.
- We print the result complex data.
- We free the memory allocated on the host and device.
- We destroy the cuFFT plan.

In math writing the operation is:

$$X_k = FFT(x_n)$$

where:

- $X_k$ is the $k^{th}$ frequency component
- $x_n$ is the $n^{th}$ input signal

## 7.3 Conclusion

In this unit, we explored the cuBLAS and cuFFT libraries provided by NVIDIA for performing basic linear algebra and fast Fourier transform operations on NVIDIA GPUs. We learned how to use these libraries in our CUDA code to perform matrix-vector multiplication and FFT operations. These libraries provide optimized routines for performing these operations on NVIDIA GPUs, allowing us to write faster and more efficient CUDA code.

# 8.0 NVVC Complier

In this unit we will take a brief look at the compilation process for a CUDA program with the NVVC compiler.

## 8.1 What is NVVC

`nvvc` is the NVIDIA CUDA compiler driver. It is a wrapper around the host compiler (e.g. `gcc`, `cl`, `clang`) and the CUDA compiler (`nvcc`). It is used to compile CUDA code and link it to the host code.

## 8.2 How to use NVVC

To compile a CUDA program with `nvcc` you can use the following command:

```bash
nvcc -o output_file input_file.cu
```

This command will compile the `input_file.cu` and create an executable called `output_file`.

## 8.3 NVVC Options

`nvcc` has many options that you can use to control the compilation process. Here are some of the most common options:

- `-o` : Set the output file name.
- `-arch` : Set the architecture for the GPU.
- `-g` : Generate debug information.
- `-G` : Generate debug information for device code.
- `-Xcompiler` : Pass options to the host compiler.
- `-Xptxas` : Pass options to the PTX assembler.
- `-Xcudafe` : Pass options to the CUDA front-end.
- `-Xnvlink` : Pass options to the linker.

For a full list of options you can use the following command:

```bash
nvcc --help
```

Flag for profiling:

```
nvcc -lineinfo
```
## 8.4 Compilation Process

The compilation process for a CUDA program with `nvcc` is as follows:

1. The host code is compiled with the host compiler.
2. The device code is compiled with the CUDA compiler.
3. The host and device code are linked together with the linker.


![nvvc](/images/17_image.png)
As we can see in the diagram above, given a file `.cu` the`nvvc` compiler separate the code into function that will be compiled using the host complier (function that are not marked or use the `__host__` marker) and function that will be compiled using the device compiler (function that are marked or use the  `__device__` or `__global__` marker). Then the host device will be compiled into the host assembly using the available host complier (i.e. `gcc`, `cl` etc.) into `x86` or `ARM` assembly code. The device code will be compiled into `PTX` assembly code. Then the `PTX` assembly code will be compiled into `SASS` assembly code. Finally the `SASS` assembly code will be linked with the host assembly code to create the final executable.

### 8.4.1 PTX Assembly

`PTX` stands for Parallel Thread Execution. It is an intermediate assembly language that is used by the CUDA compiler to generate machine code for the GPU. The `PTX` assembly code is architecture-independent and can be compiled into machine code for any GPU architecture.

### 8.4.2 SASS Assembly

`SASS` stands for Shader Assembly. It is the machine code that is generated by the CUDA compiler from the `PTX` assembly code. The `SASS` assembly code is architecture-specific and is executed by the GPU.

# 9.0 `ncu` Tools

In this unit we will take a brief look at the `ncu` tools that are provided by NVIDIA to help you analyze the performance of your CUDA code.

## 9.1 What is `ncu`

`ncu` is a command-line tool that is provided by NVIDIA to help you analyze the performance of your CUDA code. It can be used to profile your code and identify performance bottlenecks.

## 9.2 How to use `ncu`

To use `ncu` you can run the following command:

```bash
ncu -o output_file input_file
```

This command will profile the `input_file` and create a report in the `output_file`.

## 9.3 `ncu` Options

`ncu` has many options that you can use to control the profiling process. Here are some of the most common options:

- `-o` : Set the output file name.
- `-k` : Set the kernel to profile.
- `-s` : Set the number of profiling sessions.
- `-t` : Set the number of profiling iterations.
- `-f` : Set the output format.
- `-m` : Set the metrics to collect.
- `-w` : Set the working directory.

For a full list of options you can use the following command:

```bash
ncu --help
```

## 9.4 Profiling Process

The profiling process with `ncu` is as follows:

1. The `ncu` tool will run your CUDA code and collect performance data.
2. The performance data will be analyzed and a report will be generated.
3. The report will show you the performance bottlenecks in your code and suggest ways to optimize it.

![ncu](/images/18_image.png)

As we can see in the diagram above we can see how `ncu` is profiling a kernel. It will setup the environment and run the kernel multiple times to collect performance data. Then it will analyze the data and generate a report that shows the performance bottlenecks in the code.

## 9.5 Reading the Report

The `ncu` report will show you the following information:

The report will show you the following information:

- The device vector add time.
- The host vector add time.
- The DRAM frequency.
- The SM frequency.
- The memory throughput.
- The DRAM throughput.
- The L1/TEX cache throughput.
- The L2 cache throughput.
- The SM active cycles.
- The compute (SM) throughput.

The report will also show you the following information:

- The block size.
- The function cache configuration.
- The grid size.
- The registers per thread.
- The shared memory configuration.
- The driver shared memory per block.
- The dynamic shared memory per block.
- The static shared memory per block.
- The threads.
- The waves per SM.

The report will also show you the following information:

- The block limit SM block.
- The block limit registers block.
- The block limit shared mem block.
- The block limit warps block.
- The theoretical active warps per SM.
- The theoretical occupancy.
- The achieved occupancy.
- The achieved active warps per SM.

