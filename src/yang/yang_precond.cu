#include <stdio.h>
#include <stdlib.h>
#include <cmath> 
#include <chrono>
#include "gpu_pcg.cuh"
#include "utils.h"

template<typename T>
void run_benchmark(uint32_t nx, uint32_t N, const std::string& data_path) {
  //-------- setup PCG  ---------
  const uint32_t Nnx = nx * N;
  const uint32_t size_h = 3*Nnx*nx;
  struct pcg_config<T> config;
  config.pcg_max_iter = NBR_ITERATION_MAX;
  config.pcg_org_trans = false;
  config.pcg_poly_order = 1;
  config.pcg_poly_coeff[0] = 1.0;
  std::string S_path = data_path + "/S.txt";
  std::string gamma_path = data_path + "/h_gamma.txt";
  std::string h_S_path = data_path + "/h_S.txt";
  std::string h_H_path = data_path + "/H.txt";
  std::string h_P_path = data_path + "/P.txt";

  //-------- data reading  ---------
  T* h_S = (T*) calloc(size_h, sizeof(T));
  T* h_gamma = (T*) calloc(Nnx, sizeof(T));
  T* h_lambda = (T*) calloc(Nnx, sizeof(T));
  T* h_H = (T*) calloc(size_h, sizeof(T));
  T* h_Pinv = (T*) calloc(size_h, sizeof(T));
  T* S = (T*) calloc(Nnx*Nnx, sizeof(T));
  readArrayFromFile(Nnx, gamma_path.c_str(), h_gamma);
  readArrayFromFile(size_h, h_S_path.c_str(), h_S);
  readArrayFromFile(size_h, h_H_path.c_str(), h_H);
  readArrayFromFile(size_h, h_P_path.c_str(), h_Pinv);
  readArrayFromFile(Nnx*Nnx, S_path.c_str(), S);

  #if VERBOSE
    printVector("h_lambda", h_lambda, Nnx);
  #endif

  #if STATExCOMPUTER or KNOTxCOMPUTER
    auto start = std::chrono::high_resolution_clock::now();
  #endif

  //-------- Compute PCG  ---------
  float kernel_time_ms = 0;
  uint32_t nbr_iter_resolving = solvePCG<T>(h_S,
                              h_Pinv,
                              h_H,
                              h_gamma,
                              h_lambda,
                              nx,
                              N,
                              &config,
                              &kernel_time_ms);

  //-------- print benchmark result  ---------

  #if VERBOSE
    printVector("h_lambda", h_lambda, Nnx, 2);
    printVector("h_gamma", h_gamma, Nnx, 2);
    printMatrix("S", S, Nnx, 2);
    T error(0.0);
    error_L2<T>(S, h_gamma, h_lambda, Nnx, error);
    print_error(error);
  #endif

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

  #if STATExNBR_ITERATION
    std::cout << nx << std::endl;
    std::cout << nbr_iter_resolving << std::endl;
  #endif

  #if KNOTxNBR_ITERATION
    std::cout << N << std::endl;
    std::cout << nbr_iter_resolving << std::endl;
  #endif

  #if NBR_ITERATIONxERROR
    T error(0.0);
    error_L2<T>(S, h_gamma, h_lambda, Nnx, error);
    std::cout << nbr_iter_resolving << std::endl;
    print_error(error);
  #endif

  #if KERNELxERROR
    T error(0.0);
    error_L2<T>(S, h_gamma, h_lambda, Nnx, error);
    std::cout << kernel_time_ms << std::endl;
    print_error(error);
  #endif

  free(h_S);
  free(h_Pinv);
  free(h_H);
  free(h_gamma);
  free(h_lambda);
  free(S);
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