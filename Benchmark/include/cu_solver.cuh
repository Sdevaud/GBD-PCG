#pragma once

#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <iostream>
#include <type_traits>

// =======================
// Helpers d'erreur
// =======================

inline void cudaCheck(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess) {
        std::cerr << "CUDA error " << cudaGetErrorString(err)
                  << " at " << file << ":" << line << std::endl;
    }
}

inline void cusolverCheck(cusolverStatus_t status, const char* file, int line)
{
    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "cuSOLVER error " << status
                  << " at " << file << ":" << line << std::endl;
    }
}

#define CUDA_CHECK(call)      cudaCheck((call), __FILE__, __LINE__)
#define CUSOLVER_CHECK(call)  cusolverCheck((call), __FILE__, __LINE__)

// =======================
// Solver GPU (Cholesky)
// =======================

template<typename T>
void solver_gpu(const T* h_S,
                const T* h_gamma,
                T* h_lambda,
                const int* rowptr,
                const int* colind,
                const int Nnx,         // dimension de A
                const int size_S,      // nnz
                const int size_rowptr, // doit être Nnx+1
                float* kernel_time_ms)
{
    // Sanity check CSR
    if (size_rowptr != Nnx + 1) {
        std::cerr << "ERROR: size_rowptr (" << size_rowptr
                  << ") != Nnx+1 (" << (Nnx+1) << ")\n";
    }

    // --- Allocations device ---
    T *dh_S = nullptr, *dh_gamma = nullptr, *dh_lambda = nullptr;
    int *drowptr = nullptr, *dcolind = nullptr;

    CUDA_CHECK(cudaMalloc(&dh_S,      size_S      * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dh_gamma,  Nnx         * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dh_lambda, Nnx         * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&drowptr,   size_rowptr * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dcolind,   size_S      * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dh_S,     h_S,     size_S      * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dh_gamma, h_gamma, Nnx         * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(drowptr,  rowptr,  size_rowptr * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dcolind,  colind,  size_S      * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dh_lambda, 0, Nnx * sizeof(T)));

    // --- cuSOLVER handle ---
    cusolverSpHandle_t SpHandle;
    CUSOLVER_CHECK(cusolverSpCreate(&SpHandle));

    // --- cuSPARSE matrix descriptor ---
    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    int singularity = -1;

    // --- Timing ---
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    // --- Solve ---
    if constexpr (std::is_same<T,float>::value) {

        CUSOLVER_CHECK(
            cusolverSpScsrlsvchol(
                SpHandle,
                Nnx,         // dimension
                size_S,      // nnz
                descrA,
                dh_S,
                drowptr,
                dcolind,
                dh_gamma,
                1e-6f,       // tol
                0,           // reorder
                dh_lambda,
                &singularity
            )
        );

    } else if constexpr (std::is_same<T,double>::value) {

        CUSOLVER_CHECK(
            cusolverSpDcsrlsvchol(
                SpHandle,
                Nnx,
                size_S,
                descrA,
                dh_S,
                drowptr,
                dcolind,
                dh_gamma,
                1e-12,       // tol
                0,
                dh_lambda,
                &singularity
            )
        );
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(kernel_time_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    if (singularity != -1) {
        std::cerr << "[cuSOLVER] Matrix is singular at row " << singularity << std::endl;
    }

    // --- Résultat host ---
    CUDA_CHECK(cudaMemcpy(h_lambda, dh_lambda, Nnx * sizeof(T), cudaMemcpyDeviceToHost));

    // --- Cleanup ---
    CUSOLVER_CHECK(cusolverSpDestroy(SpHandle));
    cusparseDestroyMatDescr(descrA);

    cudaFree(dh_S);
    cudaFree(dh_gamma);
    cudaFree(dh_lambda);
    cudaFree(drowptr);
    cudaFree(dcolind);
}
