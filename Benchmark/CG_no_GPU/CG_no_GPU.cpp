#include <stdio.h>
#include <stdlib.h>
#include <cmath> 
#include <chrono>
#include "utils.h"
#include "CG_no_GPU.h"
#include "constant.h"

template<typename T>
void run_benchmark(uint32_t state_size, uint32_t knot_points, uint32_t& nbr_iteration) {

  const uint32_t Nnx = state_size * knot_points;
  // data generation
  T* S = (T*) calloc(Nnx*Nnx, sizeof(T));
  T* h_gamma = (T*) calloc(Nnx, sizeof(T));
  T* h_lambda = (T*) calloc(Nnx, sizeof(T));
  readArrayFromFile(Nnx, "../include/data/h_gamma.txt", h_gamma);
  readArrayFromFile(Nnx*Nnx, "../include/data/S.txt", S);
    
  // --- Start Chrono ---
  auto start = std::chrono::high_resolution_clock::now();

  Conjugate_Gradien<T>(S, h_gamma, h_lambda, state_size, knot_points, nbr_iteration);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> exec_time_ms = end - start;
  std::cout << exec_time_ms.count() << std::endl;

  #if ERROR_DOUBLE or ERROR_FLOAT
    std::cout << nbr_iteration << std::endl;
    T error(0.0);
    error_computation<T>(S, h_gamma, h_lambda, Nnx, error);
    print_error(error);
  #endif

  free(S);
  free(h_gamma);
  free(h_lambda);
}

int main() {

  const uint32_t state_size = STATE_SIZE;
  const uint32_t knot_points = KNOT_POINTS;
  uint32_t nbr_iteration = NBR_ITERATION;

  #if TIME_EXECUTION_DOUBLE or ERROR_DOUBLE
    run_benchmark<double>(state_size, knot_points, nbr_iteration);
  #endif

  #if TIME_EXECUTION_FLOAT or ERROR_FLOAT
    run_benchmark<float>(state_size, knot_points, nbr_iteration);
  #endif

  return 0;
}