#include <stdio.h>
#include <stdlib.h>
#include <cmath> 
#include <chrono>
#include <Eigen/Dense>
#include <iostream>
#include "Gauss_Jordan.cuh"
#include "generate_A_SPD.cuh"

#ifndef STATE_SIZE
#define STATE_SIZE 40
#endif

#ifndef KNOT_POINTS
#define KNOT_POINTS 100
#endif


template<typename T>
void run_benchmark(uint32_t state_size, uint32_t knot_points) {
  const int Nnx = state_size * knot_points;

  T* h_S = generate_spd_block_tridiagonal<T>(state_size, knot_points);
  T* h_gamma = generate_random_vector<T>(Nnx);

  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> M(h_S, Nnx, Nnx);
  Eigen::Map<Eigen::VectorXd> p(h_gamma, Nnx);

  auto start = std::chrono::high_resolution_clock::now();
  Eigen::VectorXd h_lambda = M.colPivHouseholderQr().solve(p);
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<T, std::milli> exec_time_ms = end - start;
  std::cout << exec_time_ms.count() << std::endl;

  free(h_S);
  free(h_gamma);
}

int main() {

  const uint32_t state_size   = STATE_SIZE;
  const uint32_t knot_points = KNOT_POINTS;

  run_benchmark<double>(state_size, knot_points);

  return 0;
}