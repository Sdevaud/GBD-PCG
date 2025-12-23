#include <stdio.h>
#include <stdlib.h>
#include <cmath> 
#include <chrono>
#include <string>
#include <iostream>
#include "cu_solver.cuh"
#include "constants.cuh"
#include "utils.h"

template<typename T>
void run_benchmark(uint32_t nx, uint32_t N, const std::string& data_path) {

  const uint32_t Nnx = nx * N;
  std::string S_path = data_path + "/S.txt";
  std::string gamma_path = data_path + "/h_gamma.txt";

  //-------- data reading  ---------
  T* S = (T*) calloc(Nnx*Nnx, sizeof(T));
  T* h_gamma = (T*) calloc(Nnx, sizeof(T));
  T* h_lambda = (T*) calloc(Nnx, sizeof(T));
  readArrayFromFile(Nnx*Nnx, S_path.c_str(), S);
  readArrayFromFile(Nnx, gamma_path.c_str(), h_gamma);
  int value_size = 0;
  int rowptr_size = 0;
  int colInd_size = 0;
  int* rowptr_S = generate_rowptr(nx, N, rowptr_size);
  int* colInd_S = generate_colind(nx, N, colInd_size, rowptr_S, rowptr_size);
  T* value_S = value<T>(S, nx, N, value_size, rowptr_S, colInd_S, rowptr_size, colInd_size);

  printMatrix<T>("S", S, Nnx);
  printVector<int>("rowptr_S", rowptr_S, rowptr_size);
  printVector<int>("colInd_S", colInd_S, colInd_size);
  printVector<T>("value_S", value_S, value_size);
  printf("rowptr_size: %d\n", rowptr_size);
  printf("colInd_size: %d\n", colInd_size);
  printf("value_size: %d\n", value_size);

  #if STATExCOMPUTER or KNOTxCOMPUTER
    auto start = std::chrono::high_resolution_clock::now();
  #endif

  //-------- Compute Ax = b  ---------
  float kernel_time_ms = 0;
  solver_gpu<T>(value_S,
                h_gamma,
                h_lambda,
                rowptr_S,
                colInd_S,
                Nnx,
                value_size,
                rowptr_size,
                &kernel_time_ms);

  //-------- print benchmark result  ---------
  T error(0.0);
  error_L2<T>(S, h_gamma, h_lambda, Nnx, error);
  print_error(error);
  printVector<T>("h_lambda", h_lambda, Nnx);
  #if STATExKERNEL
    std::cout << nx << std::endl;
    std::cout << kernel_time_ms << std::endl;
  #endif 

  #if KNOTxKERNEL
    std::cout << N << std::endl;
    std::cout << kernel_time_ms << std::endl;
  #endif

  #if STATExCOMPUTER
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> exec_time = end - start;
    std::cout << nx << std::endl;
    std::cout << exec_time.count() << std::endl;
  #endif

  #if KNOTxCOMPUTER
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> exec_time = end - start;
    std::cout << N << std::endl;
    std::cout << exec_time.count() << std::endl;
  #endif

  free(S);
  free(h_gamma);
}

int main() {

  const uint32_t state_size = STATE_SIZE;
  const uint32_t knot_points = KNOT_POINTS;
  std::string data_path = "../include/data";  // default

  #if DOUBLE
    run_benchmark<double>(state_size, knot_points, data_path);
  #else
    run_benchmark<float>(state_size, knot_points, data_path);
  #endif

  return 0;
}