#include <stdio.h>
#include <stdlib.h>
#include <cmath> 
#include <chrono>
#include <Eigen/Dense>
#include <iostream>
#include "Gauss_Jordan.cuh"
#include "generate_A_SPD.cuh"


template<typename T>
void run_benchmark(const uint32_t state_size, const uint32_t knot_points) {

  const int Nnx = state_size * knot_points;
  T* A = generate_spd_block_tridiagonal<T>(state_size, knot_points);
  T* B = generate_random_vector<T>(Nnx);

  // --- Start Chrono ---
  auto start = std::chrono::high_resolution_clock::now();

  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> M(A, Nnx, Nnx);
  Eigen::Map<Eigen::VectorXd> p(B, Nnx);
  Eigen::VectorXd x = M.colPivHouseholderQr().solve(p);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<T, std::milli> exec_time_ms = end - start;

  // --- End Chrono ---

  // print only time execution for benchmar.py (in ms)
  std::cout << exec_time_ms.count() << std::endl;

  free(A);
  free(B);
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