#include <stdio.h>
#include <stdlib.h>
#include <cmath> 
#include <chrono>
#include "utils.h"
#include "CG_no_GPU.h"
#include "constant.h"

template<typename T>
void run_benchmark(uint32_t state_size, uint32_t knot_points, uint32_t& nbr_iteration, const std::string& data_path) {

  const uint32_t Nnx = state_size * knot_points;
  std::string S_path = data_path + "/S.txt";
  std::string gamma_path = data_path + "/h_gamma.txt";

  //-------- data reading  ---------
  T* S = (T*) calloc(Nnx*Nnx, sizeof(T));
  T* h_gamma = (T*) calloc(Nnx, sizeof(T));
  T* h_lambda = (T*) calloc(Nnx, sizeof(T));
  readArrayFromFile(Nnx*Nnx, S_path.c_str(), S);
  readArrayFromFile(Nnx, gamma_path.c_str(), h_gamma);
    
  // --- Start Chrono ---
  auto start = std::chrono::high_resolution_clock::now();

  Conjugate_Gradien<T>(S, h_gamma, h_lambda, state_size, knot_points, nbr_iteration);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> exec_time_ms = end - start;
  std::cout << exec_time_ms.count() << std::endl;

  #if ERROR_DOUBLE or ERROR_FLOAT
    std::cout << nbr_iteration << std::endl;
    T error(0.0);
    error_L2<T>(S, h_gamma, h_lambda, Nnx, error);
    print_error(error);
  #endif

  free(S);
  free(h_gamma);
  free(h_lambda);
}

int main() {

  const uint32_t state_size = STATE_SIZE;
  const uint32_t knot_points = KNOT_POINTS;
  uint32_t nbr_iteration = NBR_ITERATION;
   std::string data_path = "./include/data";  // default

  #if TIME_EXECUTION_DOUBLE or ERROR_DOUBLE
    run_benchmark<double>(state_size, knot_points, nbr_iteration, data_path);
  #endif

  #if TIME_EXECUTION_FLOAT or ERROR_FLOAT
    run_benchmark<float>(state_size, knot_points, nbr_iteration, data_path);
  #endif

  return 0;
}