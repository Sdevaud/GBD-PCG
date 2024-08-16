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

    T h_lambda[Nnx];
    for (int i = 0; i < Nnx; i++) {
        h_lambda[i] = 0;
    }
    T h_gamma[Nnx];
    T *h_S, *h_Pinv;
    std::string file_name;

    if (PCG_TYPE) {
        // TRANS
        file_name = "data/H_tilde.txt";
        h_S = new T[2 * Nnx2 + Nnx];
        h_Pinv = new T[2 * Nnx2 + Nnx];
        readArrayFromFile(2 * Nnx2, "data/Pob.txt", h_Pinv + Nnx);
        readArrayFromFile(Nnx, "data/Pdb.txt", h_Pinv);
        readArrayFromFile(2 * Nnx2, "data/Sob.txt", h_S + Nnx);
        readArrayFromFile(Nnx, "data/Sdb.txt", h_S);
        readArrayFromFile(Nnx, "data/gamma_tilde.txt", h_gamma);
    } else {
        // ORG
        file_name = "data/H.txt";
        h_S = new T[3 * Nnx2];
        h_Pinv = new T[3 * Nnx2];
        readArrayFromFile(3 * Nnx2, "data/S.txt", h_S);
        readArrayFromFile(3 * Nnx2, "data/P.txt", h_Pinv);
        readArrayFromFile(Nnx, "data/gamma.txt", h_gamma);
    }


    struct pcg_config<T> config;
    config.pcg_org_trans = PCG_TYPE;
    config.pcg_poly_order = PRECOND_POLY_ORDER;
    printf("summary of PCG %s\n", PCG_TYPE ? "TRANS" : "ORG");
    printf("type of preconditioner: p%ds3\n", PRECOND_POLY_ORDER);

    if (PRECOND_POLY_ORDER > 0) {
        T h_H[3 * Nnx2];
        const char *all = file_name.c_str();
        printf("reading from file %s\n", all);
        readArrayFromFile(3 * Nnx2, all, h_H);
        int param_length = 9;

        if (PRECOND_POLY_ORDER == 1) {
            for (int i = 0; i < param_length; i++) {
                T a = 1 + i * 0.5;
                config.pcg_poly_coeff[0] = a;
                printf("a = %f\n", config.pcg_poly_coeff[0]);
                uint32_t res = solvePCG<T>(h_S,
                                           h_Pinv,
                                           h_H,
                                           h_gamma,
                                           h_lambda,
                                           state_size,
                                           knot_points,
                                           &config);
                T norm = 0;
                for (int i = 0; i < Nnx; i++) {
                    norm += h_lambda[i] * h_lambda[i];
                    h_lambda[i] = 0;
                }
                printf("result: lambda norm = %f, pcg iter = %d\n\n", sqrt(norm), res);
            }
        }

        if (PRECOND_POLY_ORDER == 2) {
            for (int i = 0; i < param_length; i++) {
                T a = 1 + i * 0.5;
                config.pcg_poly_coeff[0] = a;
                printf("a = %f\n", config.pcg_poly_coeff[0]);
                for (int j = 0; j < param_length; j++) {
                    T b = 1 + j * 0.5;
                    config.pcg_poly_coeff[1] = b;
                    printf("b = %f\n", config.pcg_poly_coeff[1]);
                    uint32_t res = solvePCG<T>(h_S,
                                               h_Pinv,
                                               h_H,
                                               h_gamma,
                                               h_lambda,
                                               state_size,
                                               knot_points,
                                               &config);
                    T norm = 0;
                    for (int i = 0; i < Nnx; i++) {
                        norm += h_lambda[i] * h_lambda[i];
                        h_lambda[i] = 0;
                    }
                    printf("result: lambda norm = %f, pcg iter = %d\n\n", sqrt(norm), res);
                }
            }
        }

    } else {
        T *h_H = NULL;
        uint32_t res = solvePCG<T>(h_S,
                                   h_Pinv,
                                   h_H,
                                   h_gamma,
                                   h_lambda,
                                   state_size,
                                   knot_points,
                                   &config);
        T norm = 0;
        for (int i = 0; i < Nnx; i++) {
            norm += h_lambda[i] * h_lambda[i];
        }

        printf("result: lambda norm = %f, pcg iter = %d\n", sqrt(norm), res);
    }

    delete (h_S);
    delete (h_Pinv);

}

int main() {

    printf("pcg example in float\n");
    pcg_solve_example<float>();

    printf("pcg example in double\n");
    pcg_solve_example<double>();

    return 0;
}

