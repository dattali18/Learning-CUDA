#include <iostream>
#include "cuda_runtime.h"
#include "time.h"


__global__ void vectorAdd(float * a, float * b, float * c, int n) {
    /*
    * This function will be called using 1 dimensional grid and 1 dimensional block.
    */

   // get the index of the current thread
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    // check if the index is within the range of the array
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

void deviceVectorAdd(float * a, float * b, float * c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}


void create_random_vector(float * vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = rand() % 100;
    }
}


void print_vector(float * vec, int n) {
    for (int i = 0; i < n; i++) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

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

void benchmark_vectorAdd() {
    int n = 1000000;
    float * a = new float[n];
    float * b = new float[n];
    float * c = new float[n];

    create_random_vector(a, n);
    create_random_vector(b, n);

    // using the time.h library to measure the time
    clock_t start, end;

    // time the device vector add
    start = clock();
    vector_add(a, b, c, n);
    end = clock();

    float milliseconds = 0;

    milliseconds = ((float)(end - start) / CLOCKS_PER_SEC) * 1000;


    std::cout << "Device vector add time: " << milliseconds << "ms" << std::endl;

    // delete the memory
    delete[] c;

    c = new float[n];

    // time the host vector add
    start = clock();
    deviceVectorAdd(a, b, c, n);
    end = clock();

    milliseconds = ((float)(end - start) / CLOCKS_PER_SEC) * 1000;

    std::cout << "Host vector add time: " << milliseconds << "ms" << std::endl;

    // free the memory

    delete[] a;
    delete[] b;
    delete[] c;
}


int main() {
    benchmark_vectorAdd();
    return 0;
}