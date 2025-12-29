#include <stdio.h>
#include <stdlib.h>
#include <cmath> 
#include <chrono>
#include "constants.h"
#include "utils.h"
#include "gato_benchmark.cuh"
using namespace sqp;
using namespace gato;

template<typename T>
void run_benchmark(uint32_t nx, uint32_t N, const std::string& data_path) {
  //-------- setup PCG  ---------
  const uint32_t Nnx = nx * N;
  const uint32_t size_h_S = 3*Nnx*nx;
  std::string S_path = data_path + "/S.txt";
  std::string gamma_path = data_path + "/h_gamma.txt";
  std::string h_S_path = data_path + "/h_S_line.txt";
  std::string h_P_path = data_path + "/P_line.txt";
  uint32_t nbr_iter_resolving = 0;
  const uint32_t nbr_iteration_max = NBR_ITERATION_MAX;
  float kernel_time_ms = 0.0f;

  //-------- data reading  ---------
  T* h_S = (T*) calloc(size_h_S, sizeof(T));
  T* h_gamma_padding = (T*) calloc(VEC_SIZE_PADDED, sizeof(T));
  T* h_lambda_padding = (T*) calloc(VEC_SIZE_PADDED, sizeof(T));
  T* h_Pinv = (T*) calloc(size_h_S, sizeof(T));
  T* S = (T*) calloc(Nnx*Nnx, sizeof(T));
  T* h_gamma = (T*) calloc(Nnx, sizeof(T));
  T* h_lambda = (T*) calloc(Nnx, sizeof(T));
  readArrayFromFile<T>(Nnx, gamma_path.c_str(), h_gamma_padding, nx);
  readArrayFromFile<T>(size_h_S, h_S_path.c_str(), h_S);
  readArrayFromFile<T>(size_h_S, h_P_path.c_str(), h_Pinv);
  readArrayFromFile<T>(Nnx*Nnx, S_path.c_str(), S);
  readArrayFromFile<T>(Nnx, gamma_path.c_str(), h_gamma);


  #if STATExCOMPUTER or KNOTxCOMPUTER
    auto start = std::chrono::high_resolution_clock::now();
  #endif
  
  //-------- Compute PCG  ---------

  solver_PCG_GATO<T, BATCH_SIZE>(
      h_lambda_padding,
      h_S,
      h_Pinv,
      h_gamma_padding,
      size_h_S,
      nbr_iteration_max,
      nbr_iter_resolving,
      &kernel_time_ms
    );

  //-------- print benchmark result  ---------

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
    TransformPadding<T>(h_lambda_padding, h_lambda, nx, Nnx);
    T error(0.0);
    error_L2<T>(S, h_gamma, h_lambda, Nnx, error);
    std::cout << nbr_iter_resolving << std::endl;
    print_error(error);
  #endif

  #if KERNELxERROR
    TransformPadding<T>(h_lambda_padding, h_lambda, nx, Nnx);
    T error(0.0);
    error_L2<T>(S, h_gamma, h_lambda, Nnx, error);
    std::cout << kernel_time_ms << std::endl;
    print_error(error);
  #endif

  free(h_lambda_padding);
  free(h_gamma_padding);
  free(S);
  free(h_S);
  free(h_gamma);
  free(h_lambda);
  free(h_Pinv);
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