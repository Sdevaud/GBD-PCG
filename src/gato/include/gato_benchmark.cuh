#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include "constants.h"
#include "cuda.cuh"
#include "pcg_gato.cuh"


template<typename T, uint32_t BatchSize>
void solver_PCG_GATO(
                      T* lambda_batch, 
                      const T* S_batch, 
                      const T* P_inv_batch, 
                      const T* gamma_batch, 
                      const uint32_t size_matrix_h,
                      const uint32_t max_pcg_iters, 
                      uint32_t& d_iterations,
                      float* kernel_time_ms) {

  T* d_lambda_batch = nullptr;
  T* d_S_batch = nullptr;
  T* d_P_inv_batch = nullptr;
  T* d_gamma_batch = nullptr;

  gpuErrchk(cudaMalloc(&d_lambda_batch, VEC_SIZE_PADDED * sizeof(T)));
  gpuErrchk(cudaMalloc(&d_gamma_batch, VEC_SIZE_PADDED * sizeof(T)));
  gpuErrchk(cudaMalloc(&d_S_batch, size_matrix_h * sizeof(T)));
  gpuErrchk(cudaMalloc(&d_P_inv_batch, size_matrix_h * sizeof(T)));

  gpuErrchk(cudaMemset(d_lambda_batch, 0, VEC_SIZE_PADDED * sizeof(T)));
  gpuErrchk(cudaMemcpy(d_gamma_batch, gamma_batch, VEC_SIZE_PADDED * sizeof(T), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_S_batch, S_batch, size_matrix_h * sizeof(T), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_P_inv_batch, P_inv_batch, size_matrix_h * sizeof(T), cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  gpuErrchk(cudaEventCreate(&start));
  gpuErrchk(cudaEventCreate(&stop));
  gpuErrchk(cudaEventRecord(start));

  solvePCGBatched<T, BatchSize>(
      d_lambda_batch,
      d_S_batch,
      d_P_inv_batch,
      d_gamma_batch,
      max_pcg_iters,
      d_iterations
  );

  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaEventRecord(stop));
  gpuErrchk(cudaEventSynchronize(stop));
  gpuErrchk(cudaEventElapsedTime(kernel_time_ms, start, stop));
  gpuErrchk(cudaEventDestroy(start));
  gpuErrchk(cudaEventDestroy(stop));

  gpuErrchk(cudaMemcpy(lambda_batch, d_lambda_batch, VEC_SIZE_PADDED * sizeof(T), cudaMemcpyDeviceToHost));

  gpuErrchk(cudaFree(d_lambda_batch));
  gpuErrchk(cudaFree(d_S_batch));
  gpuErrchk(cudaFree(d_P_inv_batch));
  gpuErrchk(cudaFree(d_gamma_batch));
}
