#include <stdio.h>
#include <stdlib.h>
#include <cmath> 
#include <chrono>
#include "gpu_pcg.cuh"
#include "utils.h"

template<typename T>
void run_benchmark(uint32_t nx, uint32_t N, const uint32_t nbr_iteration) {
  //-------- setup PCG  ---------
  const uint32_t Nnx = nx * N;
  const uint32_t size_h = 3*Nnx*nx;
  struct pcg_config<T> config;
  config.pcg_max_iter = nbr_iteration;
  config.pcg_org_trans = false;
  config.pcg_poly_order = 1;
  config.pcg_poly_coeff[0] = 1.0;

  //-------- data reading  ---------
  T* h_S = (T*) calloc(size_h, sizeof(T));
  T* h_gamma = (T*) calloc(Nnx, sizeof(T));
  T* h_lambda = (T*) calloc(Nnx, sizeof(T));
  T* h_H = (T*) calloc(size_h, sizeof(T));
  T* h_Pinv = (T*) calloc(size_h, sizeof(T));
  readArrayFromFile(Nnx, "../include/data/h_gamma.txt", h_gamma);
  readArrayFromFile(size_h, "../include/data/h_S.txt", h_S);
  readArrayFromFile(size_h, "../include/data/H.txt", h_H);
  readArrayFromFile(size_h, "../include/data/P.txt", h_Pinv);
  #if ERROR_DOUBLE or ERROR_FLOAT
    T* S = (T*) calloc(Nnx*Nnx, sizeof(T));
    readArrayFromFile(Nnx*Nnx, "../include/data/S.txt", S);
  #endif

  float kernel_time_ms = 0;
  #if TIME_EXECUTION_DOUBLE or TIME_EXECUTION_FLOAT
    auto start = std::chrono::high_resolution_clock::now();
  #endif

  //-------- Compute PCG  ---------
  uint32_t nbr_iter_resolving = solvePCG<T>(h_S,
                              h_Pinv,
                              h_H,
                              h_gamma,
                              h_lambda,
                              nx,
                              N,
                              &config,
                              &kernel_time_ms);

  //-------- print benchmark result  ---------
  #if TIME_EXECUTION_DOUBLE or TIME_EXECUTION_FLOAT
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> exec_time = end - start;
    std::cout << exec_time.count() << std::endl;
    std::cout << kernel_time_ms << std::endl;
  #endif

  #if ERROR_DOUBLE or ERROR_FLOAT
    std::cout << kernel_time_ms << std::endl;
    std::cout << nbr_iter_resolving << std::endl;
    T error(0.0);
    error_L2<T>(S, h_gamma, h_lambda, Nnx, error);
    print_error(error);
    free(S);
  #endif

  free(h_S);
  free(h_Pinv);
  free(h_H);
  free(h_gamma);
  free(h_lambda);
}

int main() {

  const uint32_t state_size = STATE_SIZE;
  const uint32_t knot_points = KNOT_POINTS;
  const uint32_t nbr_iteration = NBR_ITERATION;

  #if TIME_EXECUTION_DOUBLE or ERROR_DOUBLE
    run_benchmark<double>(state_size, knot_points, nbr_iteration);
  #endif

  #if TIME_EXECUTION_FLOAT or ERROR_FLOAT
    run_benchmark<float>(state_size, knot_points, nbr_iteration);
  #endif

  return 0;
}