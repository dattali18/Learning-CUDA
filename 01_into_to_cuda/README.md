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

