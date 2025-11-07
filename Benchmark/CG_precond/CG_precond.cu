#include <stdio.h>
#include <stdlib.h>
#include <cmath> 
#include <chrono>
#include "gpu_pcg.cuh"
#include "generate_A_SPD.cuh"

template<typename T>
void run_benchmark(const uint32_t state_size, const uint32_t knot_points) {
  // const int Nnx2 = knot_points * state_size * state_size;
  const int Nnx = state_size * knot_points;
  struct pcg_config<T> config;
  config.pcg_org_trans = false;
  config.pcg_poly_order = 0;

  // data generetion
  T* S = generate_spd_block_tridiagonal<T>(state_size, knot_points);
  T* h_gamma = generate_random_vector<T>(Nnx);
  T h_lambda[Nnx];
  for (int i = 0; i < Nnx; i++) {
      h_lambda[i] = 0.0;
  }

//   T S[] = {
//     1.957599236045914, 0.611157926864687, -1.139722050787696, -0.539741746273675, -1.001055234691629, -0.231073487454618, 0, 0, 0, 0, 0, 0,
//     0.611157926864687, 2.812541571546651, -1.714232019374743, -1.395485479323900, 0.401242601647927, 0.928599520084132, 0, 0, 0, 0, 0, 0,
//     -1.139722050787696, -1.714232019374743, 4.360909631391179, 0.185847975084240, -2.357156807905404, -3.111281776294233, 0, 0, 0, 0, 0, 0,
//     -0.539741746273675, -1.395485479323900, 0.185847975084240, 5.884079680302322, 4.504240400558649, 1.551051126038390, -1.502756618625541, -1.390912157099733, -0.941138350510598, 0, 0, 0,
//     -1.001055234691629, 0.401242601647927, -2.357156807905404, 4.504240400558648, 8.778994067261189, 4.180471780545775, -2.711943909814732, -0.607831314151821, -1.161131511623869, 0, 0, 0,
//     -0.231073487454618, 0.928599520084132, -3.111281776294233, 1.551051126038390, 4.180471780545775, 5.325209350235607, -1.473917392512167, -0.292394521081281, -1.675767434755325, 0, 0, 0,
//     0, 0, 0, -1.502756618625541, -2.711943909814732, -1.473917392512167, 6.475199980683764, 0.602430203312538, 3.660585255045144, -1.044674328320805, -0.625063584929326, -0.743170949136368,
//     0, 0, 0, -1.390912157099733, -0.607831314151821, -0.292394521081281, 0.602430203312538, 5.503791437104911, -0.078546341237862, -0.525135921027481, -1.561179281344278, 0.785684513780924,
//     0, 0, 0, -0.941138350510598, -1.161131511623869, -1.675767434755325, 3.660585255045145, -0.078546341237862, 4.602096406602015, -0.001789642805804, 0.209333229137271, -0.981853242219615,
//     0, 0, 0, 0, 0, 0, -1.044674328320805, -0.525135921027481, -0.001789642805804, 7.684929592478735, 4.348257326175189, 1.247781042185882,
//     0, 0, 0, 0, 0, 0, -0.625063584929326, -1.561179281344278, 0.209333229137271, 4.348257326175189, 6.338298752811880, 0.341315323844955,
//     0, 0, 0, 0, 0, 0, -0.743170949136368, 0.785684513780924, -0.981853242219615, 1.247781042185882, 0.341315323844956, 5.866055645203859
// };

  T* h_S = transform_matrix<T>(S, state_size, knot_points);
  
  T* h_Pinv = formPolyPreconditioner_Pinv<T>(S, knot_points, state_size);
  // printMatrix<T>("S", S, Nnx);
  // printVector<T>("h_pinv", h_Pinv, 3*state_size*state_size*knot_points);
  // T* h_H = formPolyPreconditioner_H<T>(h_Pinv, knot_points, state_size);
  // printVector<T>("h_H", h_H, 3*state_size*state_size*knot_points);
  
  

  // time computation

  #if BENCHMARK
    #if MEMPCY
      auto start = std::chrono::high_resolution_clock::now();
    #endif
  #endif

  float kernel_time_ms = 0;

  T *h_H = NULL;
  uint32_t res = solvePCG<T>(h_S,
                              h_Pinv,
                              h_H,
                              h_gamma,
                              h_lambda,
                              state_size,
                              knot_points,
                              &config,
                              &kernel_time_ms);

  #if BENCHMARK
    #if MEMPCY
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> exec_time = end - start;
      std::cout << exec_time.count() << std::endl;
    #else
      std::cout << kernel_time_ms << std::endl;
    #endif
  #endif

  free(h_S);
  free(h_gamma);
  free(h_Pinv);
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