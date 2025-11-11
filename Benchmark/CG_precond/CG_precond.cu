#include <stdio.h>
#include <stdlib.h>
#include <cmath> 
#include <chrono>
#include "gpu_pcg.cuh"
#include "generate_A_SPD.cuh"

template<typename T>
void run_benchmark(const uint32_t state_size, const uint32_t knot_points) {
  // const int Nnx2 = knot_points * state_size * state_size;
  const int Nnx = state_size * knot_points;
  struct pcg_config<T> config;
  config.pcg_org_trans = false;
  config.pcg_poly_order = 0;

  // data generetion
  T* S = generate_spd_block_tridiagonal<T>(state_size, knot_points);
  T* h_gamma = generate_random_vector<T>(Nnx);
  T h_lambda[Nnx];
  for (int i = 0; i < Nnx; i++) {
      h_lambda[i] = 0.0;
  }


  T* h_S = transform_matrix<T>(S, state_size, knot_points);
  T* h_Pinv = formPolyPreconditioner_Pinv<T>(S, knot_points, state_size);
  
  // time computation

  #if BENCHMARK
    #if MEMPCY
      auto start = std::chrono::high_resolution_clock::now();
    #endif
  #endif

  float kernel_time_ms = 0;

  T *h_H = NULL;
  uint32_t res = solvePCG<T>(h_S,
                              h_Pinv,
                              h_H,
                              h_gamma,
                              h_lambda,
                              state_size,
                              knot_points,
                              &config,
                              &kernel_time_ms);

  #if BENCHMARK
    #if MEMPCY
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> exec_time = end - start;
      std::cout << exec_time.count() << std::endl;
    #else
      std::cout << kernel_time_ms << std::endl;
    #endif
  #endif

  std::cout <<  norm_vector<T>(h_lambda, Nnx) << std::endl;

  free(h_S);
  free(h_gamma);
  free(h_Pinv);
}

int main() {

  const uint32_t state_size   = STATE_SIZE;
  const uint32_t knot_points = KNOT_POINTS;

  run_benchmark<double>(state_size, knot_points);

  return 0;
}