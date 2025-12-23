#include <stdio.h>
#include <stdlib.h>
#include <cmath> 
#include <chrono>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include "utils.h"
#include "constants.h"

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
void run_benchmark(uint32_t nx, uint32_t N, const std::string& data_path) {
  const uint32_t Nnx = N * nx;
  std::string S_path = data_path + "/S.txt";
  std::string gamma_path = data_path + "/h_gamma.txt";

  //-------- data reading  ---------
  T* S = (T*) calloc(Nnx*Nnx, sizeof(T));
  T* gamma = (T*) calloc(Nnx, sizeof(T));
  readArrayFromFile(Nnx*Nnx, S_path.c_str(), S);
  readArrayFromFile(Nnx, gamma_path.c_str(), gamma);

  //-------- Convertion format  ---------
  Eigen::SparseMatrix<T> h_S = denseToSparse<T>(S, Nnx);
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> h_gamma(gamma, Nnx);
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>> solver;

  #if STATExCOMPUTER or KNOTxCOMPUTER or STATExKERNEL or KNOTxKERNEL
    auto start = std::chrono::high_resolution_clock::now();
  #endif

  // -------- Eigen Solver  ---------
  solver.compute(h_S);
  Eigen::Matrix<T, Eigen::Dynamic, 1> h_lambda = solver.solve(h_gamma);

  //-------- print benchmark result  ---------

  #if verbose
    printVector("h_lambda", h_lambda, Nnx, 2);
    printVector("h_gamma", h_gamma, Nnx, 2);
    printMatrix("S", S, Nnx, 2);
    T error(0.0);
    error_L2<T>(S, h_gamma, h_lambda, Nnx, error);
    print_error(error);
  #endif

  #if STATExKERNEL or STATExCOMPUTER
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> exec_time = end - start;
    std::cout << nx << std::endl;
    std::cout << exec_time.count() << std::endl;
  #endif 

  #if KNOTxKERNEL or KNOTxCOMPUTER
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> exec_time = end - start;
    std::cout << N << std::endl;
    std::cout << exec_time.count() << std::endl;
  #endif

  free(S);
  free(gamma);
}

int main() {

  const uint32_t state_size = STATE_SIZE;
  const uint32_t knot_points = KNOT_POINTS;
  std::string data_path = DATA_PATH;

  #if DOUBLE
    run_benchmark<double>(state_size, knot_points, data_path);
  #else
    run_benchmark<float>(state_size, knot_points, data_path);
  #endif

  return 0;
}