#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <type_traits>
#include <cusolverSp.h>
#include <cusparse.h>
// #include "cudss.h"

inline void cudaCheck(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess) {
        std::cerr << "CUDA error " << cudaGetErrorString(err)
                  << " at " << file << ":" << line << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

inline void cudssCheck(cudssStatus_t status, const char* file, int line)
{
    if (status != CUDSS_STATUS_SUCCESS) {
        std::cerr << "cuDSS error " << (int)status
                  << " at " << file << ":" << line << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call)  cudaCheck((call), __FILE__, __LINE__)
#define CUDSS_CHECK(call) cudssCheck((call), __FILE__, __LINE__)

template<typename T>
void solver_gpu(const T* h_csrValA,
                const T* h_b,
                T* h_x,
                const int* h_csrRowPtrA,
                const int* h_csrColIndA,
                const int n,
                const int nnz,
                const int csrRowPtrSize,
                float* kernel_time_ms)
{
    if (csrRowPtrSize != n + 1) {
        std::cerr << "ERROR: csrRowPtrSize (" << csrRowPtrSize << ") != n+1 (" << (n+1) << ")\n";
        std::exit(EXIT_FAILURE);
    }
    if (h_csrRowPtrA[n] != nnz) {
        std::cerr << "ERROR: csrRowPtrA[n] (" << h_csrRowPtrA[n] << ") != nnz (" << nnz << ")\n";
        std::exit(EXIT_FAILURE);
    }

    int *d_csrRowPtrA = nullptr, *d_csrColIndA = nullptr;
    T   *d_csrValA    = nullptr;
    T   *d_b          = nullptr;
    T   *d_x          = nullptr;

    CUDA_CHECK(cudaMalloc(&d_csrRowPtrA, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csrColIndA, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csrValA,    nnz * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_b,          n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_x,          n * sizeof(T)));

    CUDA_CHECK(cudaMemcpy(d_csrRowPtrA, h_csrRowPtrA, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csrColIndA, h_csrColIndA, nnz * sizeof(int),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csrValA,    h_csrValA,    nnz * sizeof(T),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b,          h_b,          n * sizeof(T),         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_x, 0, n * sizeof(T)));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudssHandle_t handle;
    CUDSS_CHECK(cudssCreate(&handle));
    CUDSS_CHECK(cudssSetStream(handle, stream));

    cudssConfig_t solverConfig;
    cudssData_t solverData;
    CUDSS_CHECK(cudssConfigCreate(&solverConfig));
    CUDSS_CHECK(cudssDataCreate(handle, &solverData));

    cudssMatrix_t A, b, x;

    const int nrhs = 1;
    const int64_t nrows = (int64_t)n;
    const int64_t ncols = (int64_t)n;
    const int ldb = n;
    const int ldx = n;

    cudaDataType valueType;
    if constexpr (std::is_same<T, double>::value) valueType = CUDA_R_64F;
    else valueType = CUDA_R_32F;

    CUDSS_CHECK(cudssMatrixCreateDn(&b, nrows, nrhs, ldb, (void*)d_b, valueType, CUDSS_LAYOUT_COL_MAJOR));
    CUDSS_CHECK(cudssMatrixCreateDn(&x, nrows, nrhs, ldx, (void*)d_x, valueType, CUDSS_LAYOUT_COL_MAJOR));

    cudssMatrixType_t mtype = CUDSS_MTYPE_SPD;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_LOWER;
    cudssIndexBase_t base = CUDSS_BASE_ZERO;

    CUDSS_CHECK(
        cudssMatrixCreateCsr(
            &A,
            nrows, ncols,
            (int64_t)nnz,
            (void*)d_csrRowPtrA,
            nullptr,
            (void*)d_csrColIndA,
            (void*)d_csrValA,
            CUDA_R_32I,
            valueType,
            mtype,
            mview,
            base
        )
    );

    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));
    CUDA_CHECK(cudaEventRecord(evStart, stream));

    CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS,       solverConfig, solverData, A, x, b));
    CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION,  solverConfig, solverData, A, x, b));
    CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE,          solverConfig, solverData, A, x, b));

    CUDA_CHECK(cudaEventRecord(evStop, stream));
    CUDA_CHECK(cudaEventSynchronize(evStop));
    CUDA_CHECK(cudaEventElapsedTime(kernel_time_ms, evStart, evStop));
    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));

    CUDA_CHECK(cudaMemcpyAsync(h_x, d_x, n * sizeof(T), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDSS_CHECK(cudssMatrixDestroy(A));
    CUDSS_CHECK(cudssMatrixDestroy(b));
    CUDSS_CHECK(cudssMatrixDestroy(x));
    CUDSS_CHECK(cudssDataDestroy(handle, solverData));
    CUDSS_CHECK(cudssConfigDestroy(solverConfig));
    CUDSS_CHECK(cudssDestroy(handle));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFree(d_csrRowPtrA));
    CUDA_CHECK(cudaFree(d_csrColIndA));
    CUDA_CHECK(cudaFree(d_csrValA));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_x));
}
