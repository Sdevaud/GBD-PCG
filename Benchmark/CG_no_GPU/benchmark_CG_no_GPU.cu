#include <stdio.h>
#include <stdlib.h>
#include <cmath> 
#include <chrono>
#include "CG_no_GPU.cuh"
#include "generate_A_SPD.cuh"

#ifndef STATE_SIZE
#define STATE_SIZE 40
#endif

#ifndef KNOT_POINTS
#define KNOT_POINTS 100
#endif

template<typename T>
void run_benchmark(const uint32_t state_size, const uint32_t knot_points) {

  const int Nnx = state_size * knot_points;
  T* h_S = generate_spd_block_tridiagonal<T>(state_size, knot_points);
  T* h_gamma = generate_random_vector<T>(Nnx);

  T* h_lambda = (T*) calloc(Nnx, sizeof(T));
    
  // --- Start Chrono ---
  auto start = std::chrono::high_resolution_clock::now();

  Conjugate_Gradien<T>(h_S, h_gamma, h_lambda, state_size, knot_points);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<T, std::milli> exec_time_ms = end - start;

  // --- End Chrono ---

  // print only time execution for benchmar.py (in ms)
  std::cout << exec_time_ms.count() << std::endl;

  free(h_S);
  free(h_gamma);
  free(h_lambda);
}

int main() {

  const uint32_t state_size   = STATE_SIZE;
  const uint32_t knot_points = KNOT_POINTS;

  run_benchmark<double>(state_size, knot_points);

  return 0;
}