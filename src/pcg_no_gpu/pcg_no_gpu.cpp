#include <stdio.h>
#include <stdlib.h>
#include <cmath> 
#include <chrono>
#include "utils.h"
#include "pcg_no_gpu.h"
#include "constants.h"

template<typename T>
void run_benchmark(uint32_t state_size, uint32_t knot_points, const std::string& data_path) {

  const uint32_t Nnx = state_size * knot_points;
  std::string S_path = data_path + "/S.txt";
  std::string gamma_path = data_path + "/h_gamma.txt";
  uint32_t nbr_iter_resolving = 0;
  const uint32_t nbr_iteration_max = NBR_ITERATION_MAX;

  //-------- data reading  ---------
  T* S = (T*) calloc(Nnx*Nnx, sizeof(T));
  T* h_gamma = (T*) calloc(Nnx, sizeof(T));
  T* h_lambda = (T*) calloc(Nnx, sizeof(T));
  readArrayFromFile(Nnx*Nnx, S_path.c_str(), S);
  readArrayFromFile(Nnx, gamma_path.c_str(), h_gamma);

  #if STATExCOMPUTER or KNOTxCOMPUTER or STATExKERNEL or KNOTxKERNEL or KERNELxERROR
    auto start = std::chrono::high_resolution_clock::now();
  #endif

    //-------- Compute PCG  ---------
  Conjugate_Gradien<T>(S, h_gamma, h_lambda, state_size, knot_points, nbr_iter_resolving, nbr_iteration_max);

  //-------- print benchmark result  ---------

  #if VERBOSE
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
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> exec_time = end - start;
    T error(0.0);
    error_L2<T>(S, h_gamma, h_lambda, Nnx, error);
    std::cout << exec_time.count() << std::endl;
    print_error(error);
  #endif

  free(S);
  free(h_gamma);
  free(h_lambda);
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