#pragma once

#include <iostream>
#include <stdint.h>
#include "gpuassert.cuh"
#include "types.cuh"
#include "pcg_trans.cuh"


template<typename T>
uint32_t solvePCGTrans(
        T *h_Sdb,
        T *h_Sob,
        T *h_Pinvdb,
        T *h_Pinvob,
        T *h_H,
        T *h_gamma,
        T *h_lambda,
        unsigned stateSize,
        unsigned knotPoints,
        struct pcg_config<T> *config) {
    if (!config->empty_pinv)
        printf("This api can only be called with no preconditioner\n");

    const uint32_t states_sq = stateSize * stateSize;
    const uint32_t Nnx_T = stateSize * knotPoints * sizeof(T);
    const uint32_t Nnx2_T = knotPoints * states_sq * sizeof(T);

    /* Create device memory d_Sdb, d_Sob, d_Pinvdb, d_Pinvob
     * d_gamma, d_lambda, d_r, d_p,
     * d_v_temp, d_eta_new_temp
     * d_H if applicable */

    T *d_Sdb, *d_Sob, *d_gamma, *d_lambda;
    gpuErrchk(cudaMalloc(&d_Sdb, Nnx_T));
    gpuErrchk(cudaMalloc(&d_Sob, 2 * Nnx2_T));
    gpuErrchk(cudaMalloc(&d_gamma, Nnx_T));
    gpuErrchk(cudaMalloc(&d_lambda, Nnx_T));

    T *d_Pinvdb, *d_Pinvob;
    gpuErrchk(cudaMalloc(&d_Pinvdb, Nnx_T));
    gpuErrchk(cudaMalloc(&d_Pinvob, 2 * Nnx2_T));

    T *d_H;
    if (config->pcg_poly_order == 1) {
        gpuErrchk(cudaMalloc(&d_H, 3 * Nnx2_T));
    }

    /*   PCG vars   */
    T *d_r, *d_p, *d_v_temp, *d_eta_new_temp;
    gpuErrchk(cudaMalloc(&d_r, Nnx_T));
    d_p = d_r;                      // share N*nx
    gpuErrchk(cudaMalloc(&d_v_temp, knotPoints * sizeof(T)));
    d_eta_new_temp = d_v_temp;      // share N


    /* Copy S, Pinv, gamma, lambda*/
    gpuErrchk(cudaMemcpy(d_Sdb, h_Sdb, Nnx_T, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_Sob, h_Sob, 2 * Nnx2_T, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_Pinvdb, h_Pinvdb, Nnx_T, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_Pinvob, h_Pinvob, 2 * Nnx2_T, cudaMemcpyHostToDevice));
    if (config->pcg_poly_order == 1) {
        gpuErrchk(cudaMemcpy(d_H, h_H, 3 * Nnx2_T, cudaMemcpyHostToDevice));
    }
    gpuErrchk(cudaMemcpy(d_lambda, h_lambda, Nnx_T, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_gamma, h_gamma, Nnx_T, cudaMemcpyHostToDevice));


    uint32_t pcg_iters = solvePCGTransCooperativeKernel(stateSize, knotPoints,
                                                        d_Sdb,
                                                        d_Sob,
                                                        d_Pinvdb,
                                                        d_Pinvob,
                                                        d_H,
                                                        d_gamma,
                                                        d_lambda,
                                                        d_r,
                                                        d_p,
                                                        d_v_temp,
                                                        d_eta_new_temp,
                                                        config);

    /* Copy data back */
    gpuErrchk(cudaMemcpy(h_lambda, d_lambda, Nnx_T, cudaMemcpyDeviceToHost));

    cudaFree(d_lambda);
    cudaFree(d_Sdb);
    cudaFree(d_Sob);
    cudaFree(d_gamma);
    cudaFree(d_Pinvdb);
    cudaFree(d_Pinvob);
    if (config->pcg_poly_order == 1) {
        cudaFree(d_H);
    }
    cudaFree(d_r);
    cudaFree(d_v_temp);

    return pcg_iters;
}


template<typename T>
uint32_t solvePCGTransCooperativeKernel(const uint32_t state_size,
                                        const uint32_t knot_points,
                                        T *d_Sdb,
                                        T *d_Sob,
                                        T *d_Pinvdb,
                                        T *d_Pinvob,
                                        T *d_H,
                                        T *d_gamma,
                                        T *d_lambda,
                                        T *d_r,
                                        T *d_p,
                                        T *d_v_temp,
                                        T *d_eta_new_temp,
                                        struct pcg_config<T> *config) {
    uint32_t *d_pcg_iters;
    gpuErrchk(cudaMalloc(&d_pcg_iters, sizeof(uint32_t)));
    bool *d_pcg_exit;
    gpuErrchk(cudaMalloc(&d_pcg_exit, sizeof(bool)));

    void *pcg_kernel = (void *) pcgTrans<T, STATE_SIZE, KNOT_POINTS>;

    // the following shall be turned off for speed
    bool gpu_check = checkPcgTransOccupancy<T>(pcg_kernel, config->pcg_block, state_size, knot_points,
                                               config->pcg_poly_order);
    // gpu_check shall always be true, o.w. the program exits
    // gpu_check true means
    //      1. Device supports Cooperative Threads
    //      2. Device has enough shared memory for the current state_size & knot_points

    void *kernelArgs[] = {
            (void *) &d_Sdb,
            (void *) &d_Sob,
            (void *) &d_Pinvdb,
            (void *) &d_Pinvob,
            (void *) &d_H,
            (void *) &d_gamma,
            (void *) &d_lambda,
            (void *) &d_r,
            (void *) &d_p,
            (void *) &d_v_temp,
            (void *) &d_eta_new_temp,
            (void *) &d_pcg_iters,
            (void *) &d_pcg_exit,
            (void *) &config->pcg_max_iter,
            (void *) &config->pcg_exit_tol,
            (void *) &config->pcg_poly_order
    };
    uint32_t h_pcg_iters;

    size_t ppcg_kernel_smem_size = pcgTransSharedMemSize<T>(state_size, knot_points, config->pcg_poly_order);

    gpuErrchk(cudaLaunchCooperativeKernel(pcg_kernel, knot_points, pcg_constants::DEFAULT_BLOCK, kernelArgs,
                                          ppcg_kernel_smem_size));
    gpuErrchk(cudaMemcpy(&h_pcg_iters, d_pcg_iters, sizeof(uint32_t), cudaMemcpyDeviceToHost));


    gpuErrchk(cudaFree(d_pcg_iters));
    gpuErrchk(cudaFree(d_pcg_exit));

    return h_pcg_iters;
}