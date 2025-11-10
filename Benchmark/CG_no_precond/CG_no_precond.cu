#include <stdio.h>
#include <stdlib.h>
#include <cmath> 
#include <chrono>
#include "gpu_pcg.cuh"
#include "generate_A_SPD.cuh"

template<typename T>
void run_benchmark(const uint32_t state_size, const uint32_t knot_points) {
  const int Nnx = state_size * knot_points;
  struct pcg_config<T> config;
  config.pcg_org_trans = false;
  config.pcg_poly_order = 0;

  // // data generetion
  // T* S = generate_spd_block_tridiagonal<T>(state_size, knot_points, 5);
  // T* h_gamma = generate_random_vector<T>(Nnx, 5);
  T h_lambda[Nnx];
  for (int i = 0; i < Nnx; i++) {
      h_lambda[i] = 0.0;
  }
  // T* h_S = transform_matrix<T>(S, state_size, knot_points);

  T h_S[] = {0,0,0,0,0,0,0,0,0,
1.24569733836824,-0.0341608523105336,-0.08643543089113,-0.0341608523105335,1.32451044904429,-0.0353592685160534,-0.0864354308911298,-0.0353592685160532,1.25808571358917,
-0.133504756587631,-0.640740042322416,-0.527761170314272,-0.461103639651493,-0.712579345050847,-0.828759690353756,-1.00836010095762,-0.0511948982126582,-1.15603651664238,
-0.133504756587631,-0.461103639651493,-1.00836010095762,-0.640740042322416,-0.712579345050847,-0.0511948982126582,-0.527761170314272,-0.828759690353756,-1.15603651664238,
5.39622341259055,0.138813226574523,1.91152558580477,0.138813226574523,3.41901814895606,3.57948020111595,1.91152558580477,3.57948020111595,9.22362713183952,
-4.82166838643697,-2.44243777779541,-7.39566807380996,-4.22884304796642,-3.6175114938964,-9.57478616939702,-3.78114911919826,-3.26270016565451,-8.60128696243932,
-4.82166838643697,-4.22884304796642,-3.78114911919826,-2.44243777779541,-3.6175114938964,-3.26270016565451,-7.39566807380996,-9.57478616939702,-8.60128696243932,
14.3232451393651,15.4881899933215,12.1820734419505,15.4881899933215,20.3860389982206,14.4518455518156,12.1820734419505,14.4518455518156,14.5289955096469,
-0.576491213127571,-0.396055206972733,-1.38652625730861,-1.00155651410375,-1.62986369711243,-0.140647936032963,-0.685742723282777,-0.454686685894559,-1.42489291464753,
-0.576491213127571,-1.00155651410375,-0.685742723282777,-0.396055206972733,-1.62986369711243,-0.454686685894559,-1.38652625730861,-0.140647936032963,-1.42489291464753,
3.19396199208879,0.826359387396342,1.62447947751673,0.826359387396342,3.25870238507693,0.887055540727173,1.62447947751673,0.887055540727173,2.97844939120032,
0,0,0,0,0,0,0,0,0};

  T h_gamma[] = {0.894364927082967,
0.361839180211903,
0.203689130268915,
0.530281080664476,
0.92720320120777,
0.209948828518869,
0.982018434403266,
0.416363845353053,
0.268663446255055,
0.730739264353682,
0.534110028726556,
0.375113415580673};

  // time computation

  #if BENCHMARK
    #if MEMPCY
      auto start = std::chrono::high_resolution_clock::now();
    #endif
  #endif

  float kernel_time_ms = 0;
  uint32_t res = solvePCGNew<T>(h_S,
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

  // std::cout << "Norm of solution vector: " << norm_vector<T>(h_lambda, Nnx) << std::endl;
  // printMatrix<T>("S", S, Nnx);
  // printVector<T>("h_S", h_S, 3*Nnx*state_size);
  printVector<T>("h_lambda", h_lambda, Nnx);


  // free(h_S);
  // free(h_gamma);
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
      std::cerr << "Usage: " << argv[0] << " <state> <horizon>" << std::endl;
      return 1;
  }

  const uint32_t state_size   = std::atoi(argv[1]);
  const uint32_t knot_points = std::atoi(argv[2]);

  run_benchmark<double>(3, 4);

  return 0;
}