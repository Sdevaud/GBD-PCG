#include <stdio.h>
#include <stdlib.h>
#include <cmath> 
#include <chrono>
#include "gpu_pcg.cuh"
#include "generate_A_SPD.cuh"

template<typename T>
void run_benchmark(const uint32_t state_size, const uint32_t knot_points) {
  const int Nnx = state_size * knot_points;
  struct pcg_config<T> config;
  config.pcg_org_trans = false;
  config.pcg_poly_order = 0;

  // data generetion
  T* S = generate_spd_block_tridiagonal<T>(state_size, knot_points, 5);
  T* h_gamma = generate_random_vector<T>(Nnx, 5);
  T h_lambda[Nnx];
  for (int i = 0; i < Nnx; i++) {
      h_lambda[i] = 0.0;
  }
  T* h_S = transform_matrix<T>(S, state_size, knot_points);

  // time computation

  #if BENCHMARK
    #if MEMPCY
      auto start = std::chrono::high_resolution_clock::now();
    #endif
  #endif

  float kernel_time_ms = 0;
  std::cout << state_size << std::endl;

  // uint32_t res = solvePCGNew<T>(h_S,
  //                           h_gamma,
  //                           h_lambda,
  //                           state_size,
  //                           knot_points,
  //                           &config,
  //                           &kernel_time_ms);

  #if BENCHMARK
    #if MEMPCY
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> exec_time = end - start;
      std::cout << exec_time.count() << std::endl;
    #else
      std::cout << kernel_time_ms << std::endl;
    #endif
  #endif

  // std::cout << "Norm of solution vector: " << norm_vector<T>(h_lambda, Nnx) << std::endl;
  printMatrix<T>("S", S, Nnx);
  printVector<T>("h_S", h_S, 3*Nnx*state_size);
  // printVector<T>("h_lambda", h_lambda, Nnx);

  free(h_S);
  free(h_gamma);
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
      std::cerr << "Usage: " << argv[0] << " <state> <horizon>" << std::endl;
      return 1;
  }

  const uint32_t state_size   = std::atoi(argv[1]);
  const uint32_t knot_points = std::atoi(argv[2]);

  run_benchmark<double>(state_size, knot_points);

  return 0;
}