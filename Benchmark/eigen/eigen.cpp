#include <stdio.h>
#include <stdlib.h>
#include <cmath> 
#include <chrono>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include "utils.h"
#include "constant.h"

template <typename T>
Eigen::SparseMatrix<T> denseToSparse(const T* dense, uint32_t n)
{
    Eigen::SparseMatrix<T> A_sparse(n, n);
    std::vector<Eigen::Triplet<T>> triplets;
    triplets.reserve(n * 10);

    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < n; j++) {
            T val = dense[i * n + j];
            if (val != T(0)) {
                triplets.emplace_back(i, j, val);
            }
        }
    }

    A_sparse.setFromTriplets(triplets.begin(), triplets.end());
    return A_sparse;
}

template<typename T>
void run_benchmark(uint32_t nx, uint32_t N) {
  const uint32_t Nnx = N * nx;

  //-------- data reading  ---------
  T* S = (T*) calloc(Nnx*Nnx, sizeof(T));
  T* h_gamma = (T*) calloc(Nnx, sizeof(T));
  readArrayFromFile(Nnx*Nnx, "../include/data/S.txt", S);
  readArrayFromFile(Nnx, "../include/data/h_gamma.txt", h_gamma);

  //-------- Convertion format  ---------
  Eigen::SparseMatrix<T> A_sparse = denseToSparse<T>(S, Nnx);
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> b(h_gamma, Nnx);
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>> solver;

  auto start = std::chrono::high_resolution_clock::now();

  // -------- Eigen Solver  ---------
  solver.compute(A_sparse);
  Eigen::Matrix<T, Eigen::Dynamic, 1> x = solver.solve(b);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<T, std::milli> exec_time_ms = end - start;
  std::cout << exec_time_ms.count() << std::endl;

  #if ERROR_DOUBLE or ERROR_FLOAT
    T error(0.0);
    error_L2<T>(S, h_gamma, x.data(), Nnx, error);
    print_error(error);
  #endif

  free(S);
  free(h_gamma);
}

int main() {

  const uint32_t state_size = STATE_SIZE;
  const uint32_t knot_points = KNOT_POINTS;

  #if TIME_EXECUTION_DOUBLE or ERROR_DOUBLE
    run_benchmark<double>(state_size, knot_points);
  #endif

  #if TIME_EXECUTION_FLOAT or ERROR_FLOAT
    run_benchmark<float>(state_size, knot_points);
  #endif
  return 0;
}