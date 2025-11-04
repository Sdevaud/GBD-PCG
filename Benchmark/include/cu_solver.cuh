#include <stdio.h>
#include <stdlib.h>
#include <cmath> 
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <iostream>
#include <vector>


template<typename T>
void solver_gpu(T* A, T* b, const int N, float* kernel_time_ms)
{
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);

    // device copies
    T *d_A, *d_b;
    cudaMalloc(&d_A, N*N*sizeof(T));
    cudaMalloc(&d_b, N*sizeof(T));

    cudaMemcpy(d_A, A, N*N*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N*sizeof(T), cudaMemcpyHostToDevice);

    int work_size = 0;
    cusolverDnDgetrf_bufferSize(handle, N, N, d_A, N, &work_size);

    T* work;
    int *devIpiv, *devInfo;
    cudaMalloc(&work, work_size * sizeof(T));
    cudaMalloc(&devIpiv, N * sizeof(int));
    cudaMalloc(&devInfo, sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Step 1: LU factorization (A = L*U)
    cusolverDnDgetrf(handle, N, N, d_A, N, work, devIpiv, devInfo);

    // Step 2: Solve A*x = b using the LU factors
    cusolverDnDgetrs(handle, CUBLAS_OP_N, N, 1, d_A, N, devIpiv, d_b, N, devInfo);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(kernel_time_ms, start, stop);

    // Copy solution back
    std::vector<T> x(N);
    cudaMemcpy(x.data(), d_b, N*sizeof(T), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(work);
    cudaFree(devIpiv);
    cudaFree(devInfo);
    cusolverDnDestroy(handle);
}