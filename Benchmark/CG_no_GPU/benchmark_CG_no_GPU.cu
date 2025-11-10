#include <stdio.h>
#include <stdlib.h>
#include <cmath> 
#include <chrono>
#include "CG_no_GPU.cuh"
#include "generate_A_SPD.cuh"

template<typename T>
void run_benchmark(const uint32_t state_size, const uint32_t knot_points) {

  const int Nnx = state_size * knot_points;
  T* A = generate_spd_block_tridiagonal<T>(state_size, knot_points, 5);
  T* B = generate_random_vector<T>(Nnx, 5);

  T* C = (double*)malloc(Nnx * sizeof(double));
  for (int i = 0; i < Nnx; ++i) C[i] = 0.0;
    
  // --- Start Chrono ---
  auto start = std::chrono::high_resolution_clock::now();

  Conjugate_Gradien<T>(A, B, C, state_size, knot_points);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<T, std::milli> exec_time_ms = end - start;

  // --- End Chrono ---

  // print only time execution for benchmar.py (in ms)
  std::cout << exec_time_ms.count() << std::endl;
  std::cout << "Norm of solution vector: " << norm_vector<T>(C, Nnx) << std::endl;

  printVector<T>("h_lambda", C, Nnx);

  free(A);
  free(B);
  free(C);
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