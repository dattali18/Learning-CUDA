# 0. Intro to GPU

In This unit, we will explore the microarchitecture of GPU and how it differs from CPU.

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


