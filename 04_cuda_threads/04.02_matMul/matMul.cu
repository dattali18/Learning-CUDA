#include "stdio.h"
#include "assert.h"

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"


// helper functions
#include "helper_cuda.h"
#include "helper_functions.h"

/**
 * In this file we will demonstreate a kernel for matrix multiplication
 * Also this kernel is using template
 */

/**
 * @brief Kernel for matrix multiplication
 * The kernel will perform A * B = C
 * The matrix A is of size (wA x hA)
 * This matrix is stored in row-major order which means that the elements are stored in the following way
 * A[0] A[1] A[2] ... A[wA-1] A[wA] A[wA+1] ... A[2*wA-1] ... A[wA*hA-1]
 */
template <int BLOCK_SIZE> __global__ void matMul(float * A, float * B, float * C, int wA, int wB) {
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

    // Csub is used to store the element of the block sub-matrix that is computed by the thread
    float Csub = 0.0f;

    // starting the algorithm

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

    // write the block sub-matrix to global memory
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

void RandomInit(float * data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = rand() / (float)RAND_MAX;
    }
}

// host function to launch the kernel
void MatrixMultiply(int argc, char **argv, int block_size, const dim3 &dimsA, const dim3 &dimsB) {
    // allocate memory on the host
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;

    // in this example we will show a new function called cudaMallocHost that will allocate memory on the host
    float * h_A;

    checkCudaErrors(cudaMallocHost((void **)&h_A, mem_size_A));

    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;

    float * h_B;

    checkCudaErrors(cudaMallocHost((void **)&h_B, mem_size_B));

    //  we will introduce a new concept called stream in CUDA
    // we will talk about streams in the in the section
    cudaStream_t stream;


    // initialize the input data
    RandomInit(h_A, size_A);
    RandomInit(h_B, size_B);

    // allocate memory on the device
    float * d_A;
    float * d_B;
    float * d_C;

    checkCudaErrors(cudaMalloc((void **)&d_A, mem_size_A));
    checkCudaErrors(cudaMalloc((void **)&d_B, mem_size_B));

    // copy the host memory to the device
    checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));

    // allocate memory for the result on the host
    dim3 dimsC(dimsB.x, dimsA.y, 1);

    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float * h_C;

    checkCudaErrors(cudaMallocHost((void **)&h_C, mem_size_C));

    // allocate memory for the result on the device
    float * d_C;

    checkCudaErrors(cudaMalloc((void **)&d_C, mem_size_C));

    // setup execution parameters
    dim3 threads(block_size, block_size);

    // we will introduce a new concept called grid in CUDA
    // we will talk about grid in the section
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    // create and start the timer
    StopWatchInterface * timer = NULL;

    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // execute the kernel   
    matMul<block_size><<<grid, threads>>>(d_A, d_B, d_C, dimsA.x, dimsB.x);

    // check if kernel execution generated an error
    getLastCudaError("Kernel execution failed");

    // copy result from device to host
    checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));

    // stop and destroy the timer
    sdkStopTimer(&timer);

    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

    // clean up memory
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    checkCudaErrors(cudaFreeHost(h_A));
    checkCudaErrors(cudaFreeHost(h_B));
    checkCudaErrors(cudaFreeHost(h_C));

    // clean up stream
    checkCudaErrors(cudaStreamDestroy(stream));

    // reset the device
    checkCudaErrors(cudaDeviceReset());

    printf("Done\n");
}

int main(int argc, char **argv) {
    // set the block size
    int block_size = 16;

    // set the matrix size
    int size = 16;

    // set the matrix dimensions
    dim3 dimsA(size * 2, size * 2, 1);
    dim3 dimsB(size * 2, size * 2, 1);

    // execute the kernel
    MatrixMultiply(argc, argv, block_size, dimsA, dimsB);

    return 0;
}