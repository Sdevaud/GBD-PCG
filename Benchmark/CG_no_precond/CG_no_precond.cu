#include <stdio.h>
#include <stdlib.h>
#include <cmath> 
#include <chrono>
#include "gpu_pcg.cuh"
#include "generate_A_SPD.cuh"

template<typename T>
void run_benchmark(uint32_t state_size, uint32_t knot_points, unsigned int random = 0, const uint32_t nbr_iteration = 10) {
  const uint32_t Nnx = state_size * knot_points;
  struct pcg_config<T> config;
  config.pcg_max_iter = nbr_iteration;
  config.pcg_org_trans = false;
  config.pcg_poly_order = 0;

  // data generation
  T* S = generate_spd_block_tridiagonal<T>(state_size, knot_points, random);
  T* h_gamma = generate_random_vector<T>(Nnx, random);
  T* h_lambda = (T*) calloc(Nnx, sizeof(T));
  T* h_S = transform_matrix<T>(S, state_size, knot_points);

  // time computation
  float kernel_time_ms = 0;
  #if TIME_EXECUTION_DOUBLE or TIME_EXECUTION_FLOAT
    auto start = std::chrono::high_resolution_clock::now();
  #endif
  
  uint32_t res = solvePCGNew<T>(h_S,
                            h_gamma,
                            h_lambda,
                            state_size,
                            knot_points,
                            &config,
                            &kernel_time_ms);


  #if TIME_EXECUTION_DOUBLE or TIME_EXECUTION_FLOAT
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> exec_time = end - start;
    std::cout << exec_time.count() << std::endl;
    std::cout << kernel_time_ms << std::endl;
  #endif

  #if ERROR_DOUBLE or ERROR_FLOAT
    T error(0.0);
    error_computation<T>(S, h_gamma, h_lambda, Nnx, error);
    print_error(error);
    std::cout << nbr_iteration << std::endl << kernel_time_ms << std::endl;
  #endif

  #if DEBUG
    printMatrix("S", S, Nnx);
    printVector("h_gamma", h_gamma, Nnx);
    printVector("h_lambda", h_lambda, Nnx);
    T* Axb = (T*)calloc(Nnx, sizeof(T));
    mat_mul_vector(S, h_lambda, Axb, Nnx);
    printVector("S x h_lambda", Axb, Nnx);
    bool test = is_spd(S, Nnx);
    if (test) std::cout << "SPD \n";
    else std::cout << "no SPD \n";
  #endif

  free(S);
  free(h_S);
  free(h_gamma);
  free(h_lambda);
}

int main() {

  const uint32_t state_size = STATE_SIZE;
  const uint32_t knot_points = KNOT_POINTS;
  const uint32_t nbr_iteration = NBR_ITERATION;

  #if TIME_EXECUTION_DOUBLE
    run_benchmark<double>(state_size, knot_points, 0, nbr_iteration);
  #endif

  #if TIME_EXECUTION_FLOAT
    run_benchmark<float>(state_size, knot_points, 0, nbr_iteration);
  #endif

  #if ERROR_DOUBLE
    run_benchmark<double>(state_size, knot_points, 5, nbr_iteration);
  #endif

  #if ERROR_FLOAT
    run_benchmark<float>(state_size, knot_points, 5, nbr_iteration);
  #endif 

  return 0;
}