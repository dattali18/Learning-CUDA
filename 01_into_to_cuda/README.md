# 1. Intro to CUDA

In this unit, we will explore CUDA what is it and what problem it solves.

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



