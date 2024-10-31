# CUDA Libraries

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

## cuBLAS

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


## cuFFT

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

## Conclusion

In this unit, we explored the cuBLAS and cuFFT libraries provided by NVIDIA for performing basic linear algebra and fast Fourier transform operations on NVIDIA GPUs. We learned how to use these libraries in our CUDA code to perform matrix-vector multiplication and FFT operations. These libraries provide optimized routines for performing these operations on NVIDIA GPUs, allowing us to write faster and more efficient CUDA code.

