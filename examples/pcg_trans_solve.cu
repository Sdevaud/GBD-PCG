#include <iostream>
#include <stdio.h>
#include "gpu_pcg.cuh"
#include "gpuassert.cuh"
#include "read_array.h"
#include <ctime>

#define tic      double tic_t = clock();
#define toc      std::cout << (clock() - tic_t)/CLOCKS_PER_SEC \
                           << " seconds" << std::endl;

int main(int argc, char *argv[]) {

    const uint32_t state_size = STATE_SIZE;
    const uint32_t knot_points = KNOT_POINTS;
    const int matrix_size = 2 * knot_points * state_size * state_size;
    const int vector_size = state_size * knot_points;

    float h_Pinvob[matrix_size];
    float h_Pinvdb[vector_size];
    float h_Sob[matrix_size];
    float h_Sdb[vector_size];

    readArrayFromFile(matrix_size, "data/Pob.txt", h_Pinvob);
    readArrayFromFile(vector_size, "data/Pdb.txt", h_Pinvdb);
    readArrayFromFile(matrix_size, "data/Sob.txt", h_Sob);
    readArrayFromFile(vector_size, "data/Sdb.txt", h_Sdb);

    float h_gamma[vector_size];
    readArrayFromFile(vector_size, "data/gamma_tilde.txt", h_gamma);
    float h_lambda[vector_size];
    for (int i = 0; i < vector_size; i++) {
        h_lambda[i] = 0;
    }

    struct pcg_config<float> config;
    config.pcg_poly_order = PRECOND_POLY_ORDER;
    if (PRECOND_POLY_ORDER == 1) {
        const int matrixH_size = 3 * knot_points * state_size * state_size;
        float h_H[matrixH_size];

        // information of alpha should match with MATLAB file
        int alpha_length = 9;
        float alpha_array[alpha_length];
        for (int i = 0; i < alpha_length; i++) {
            alpha_array[i] = 1 + i * 0.5;
        }

        for (int i = 0; i < alpha_length; i++) {
            float alpha = alpha_array[i];
            std::string file_name = "data/I_H_tilde_";
            file_name = file_name + std::to_string(i + 1) + ".txt";
            const char *all = file_name.c_str();
            printf("reading from file %s\n", all);
            readArrayFromFile(matrixH_size, all, h_H);
            uint32_t res = solvePCGTrans<float>(h_Sdb,
                                                h_Sob,
                                                h_Pinvdb,
                                                h_Pinvob,
                                                h_H,
                                                h_gamma,
                                                h_lambda,
                                                state_size,
                                                knot_points,
                                                &config);
            float norm = 0;
            for (int i = 0; i < vector_size; i++) {
                norm += h_lambda[i] * h_lambda[i];
                h_lambda[i] = 0;
            }
            printf("summary of PCG TRANS\n");
            printf("type of preconditioner: %s\n", PRECOND_POLY_ORDER == 1 ? "p1s3" : "p0s3");
            printf("alpha = %f\n", alpha);
            printf("result: lambda norm = %f, pcg iter = %d\n\n", sqrt(norm), res);
        }
    } else if (PRECOND_POLY_ORDER == 0) {
        float *h_H = NULL;
        uint32_t res = solvePCGTrans<float>(h_Sdb,
                                            h_Sob,
                                            h_Pinvdb,
                                            h_Pinvob,
                                            h_H,
                                            h_gamma,
                                            h_lambda,
                                            state_size,
                                            knot_points,
                                            &config);
        float norm = 0;
        for (int i = 0; i < vector_size; i++) {
            norm += h_lambda[i] * h_lambda[i];
        }
        printf("summary of PCG TRANS\n");
        printf("type of preconditioner: %s\n", PRECOND_POLY_ORDER == 1 ? "p1s3" : "p0s3");
        printf("result: lambda norm = %f, pcg iter = %d\n", sqrt(norm), res);
    }

    return 0;
}

