#include <iostream>
#include <stdio.h>
#include "gpu_pcg.cuh"
#include "gpuassert.cuh"
#include "read_array.h"
#include <ctime>

#define tic      double tic_t = clock();
#define toc      std::cout << (clock() - tic_t)/CLOCKS_PER_SEC \
                           << " seconds" << std::endl;

template<typename T>
void pcg_solve_example() {
    const uint32_t state_size = STATE_SIZE;
    const uint32_t knot_points = KNOT_POINTS;
    const int Nnx2 = knot_points * state_size * state_size;
    const int Nnx = state_size * knot_points;
    float time = 0.0;

    T h_lambda[Nnx];
    for (int i = 0; i < Nnx; i++) {
        h_lambda[i] = 0;
    }
    T h_gamma[Nnx];
    T *h_S;
    h_S = new T[3 * Nnx2];
    std::string file_name;
    readArrayFromFile(3 * Nnx2, "data/S.txt", h_S);
    readArrayFromFile(Nnx, "data/gamma.txt", h_gamma);

    struct pcg_config<T> config;
    config.pcg_org_trans = PCG_TYPE;
    config.pcg_poly_order = PRECOND_POLY_ORDER;
    printf("summary of PCG %s\n", PCG_TYPE ? "TRANS" : "ORG");
    printf("type of preconditioner: p%ds3\n", PRECOND_POLY_ORDER);

    if (PRECOND_POLY_ORDER == 1) {
        config.pcg_poly_coeff[0] = 1.0;
        printf("a1 = %f\n", config.pcg_poly_coeff[0]);
    }
    float kernel_time_ms = 0;
    uint32_t res = solvePCGNew<T>(h_S,
                                  h_gamma,
                                  h_lambda,
                                  state_size,
                                  knot_points,
                                  &config, 
                                  &kernel_time_ms);
    T norm = 0;
    for (int i = 0; i < Nnx; i++) {
        norm += h_lambda[i] * h_lambda[i];
        h_lambda[i] = 0;
    }
    printf("result: lambda norm = %f, pcg iter = %d\n", sqrt(norm), res);

    delete (h_S);

}

int main() {

//    printf("pcg example in float\n");
//    pcg_solve_example<float>();

    printf("pcg example in double\n");
    pcg_solve_example<double>();

    return 0;
}

