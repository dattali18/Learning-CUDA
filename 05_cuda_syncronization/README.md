# CUDA Synchronization

In this unit, we will explore the CUDA API for thread synchronization. We will learn how to use the different synchronization primitives provided by CUDA to write efficient and correct parallel code. We will also touch on the async operations in CUDA and the usage of atomic operations.

In CUDA there is two ways of doing things:

- **Synchronous**: The CPU waits for the GPU to finish the task.
- **Asynchronous**: The CPU does not wait for the GPU to finish the task.

## Synchronous Operations

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

## Asynchronous Operations

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

## Atomic Operations

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

## Thread Synchronisation

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


## Conclusion

In this unit, we explored the CUDA API for thread synchronization. We learned how to use the different synchronization primitives provided by CUDA to write efficient and correct parallel code. We also touched on the async operations in CUDA and the usage of atomic operations.
