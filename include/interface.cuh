#pragma once

#include <iostream>
#include <stdint.h>
#include "gpuassert.cuh"
#include "types.cuh"
#include "pcg.cuh"
#include "precondition.cuh"

template<typename T>
void recover_lamdba(uint32_t state_size, uint32_t knot_points,
                    T *d_T, T *d_lambda) {

    const uint32_t s_temp_size = sizeof(T) * (state_size * (state_size + 1) / 2 + 2 * state_size);
    recover_lambda_kernel<<<knot_points, pcg_constants::DEFAULT_BLOCK, s_temp_size>>>(state_size, knot_points,
                                                                                      d_T, d_lambda);
}

/* solvePCG is the interface function to be called from the host side (all data are declared within CPU, hence h_*)
 * it computes the preconditioners and solves the linear equation S * lambda = gamma using Preconditioned Conjugate Gradient (PCG) method
 * h_S          -> matrix, LHS of the linear equation, symmetric positive definite block tri-diagonal
 * h_gamma      -> vector, RHS of the linear equation
 * h_lambda     -> vector, placeholder for the initial guess and the final result
 * stateSize    -> integer, size of the block matrix. For OCP QP, this is nx.
 * knotPoints   -> integer, size of block rows. For OCP QP, this is N, the horizon length.
 * config       -> configuration of the PCG algorithm
 */

template<typename T>
uint32_t solvePCGNew(
        T *h_S,         // it is assumed that h_S always contains 3 * Nnx2_T elements
        T *h_gamma,
        T *h_lambda,
        unsigned stateSize,
        unsigned knotPoints,
        struct pcg_config<T> *config) {

    const uint32_t states_sq = stateSize * stateSize;
    const uint32_t Nnx_T = stateSize * knotPoints * sizeof(T);
    const uint32_t Nnx2_T = knotPoints * states_sq * sizeof(T);
    const uint32_t triangular_state = (stateSize + 1) * stateSize / 2;

    /* Create device memory d_S, d_Pinv,
     * d_gamma, d_lambda, d_r, d_p,
     * d_v_temp, d_eta_new_temp
     * d_H if applicable */

    T *d_S_in, *d_Pinv, *d_gamma, *d_lambda;
    T *d_S_out = NULL;
    T *d_T = NULL;
    T *d_H = NULL;
    gpuErrchk(cudaMalloc(&d_S_in, 3 * Nnx2_T));
    if (config->pcg_org_trans) {
        gpuErrchk(cudaMalloc(&d_S_out, 2 * Nnx2_T + Nnx_T));
        gpuErrchk(cudaMalloc(&d_Pinv, 2 * Nnx2_T + Nnx_T));
        gpuErrchk(cudaMalloc(&d_T, triangular_state * knotPoints * sizeof(T)));
    } else {
        d_S_out = d_S_in;
        gpuErrchk(cudaMalloc(&d_Pinv, 3 * Nnx2_T));
    }
    gpuErrchk(cudaMalloc(&d_lambda, Nnx_T));
    gpuErrchk(cudaMalloc(&d_gamma, Nnx_T));

    if (config->pcg_poly_order > 0) {
        gpuErrchk(cudaMalloc(&d_H, 3 * Nnx2_T));
    }

    /*   PCG vars   */
    T *d_r, *d_p, *d_v_temp, *d_eta_new_temp;
    gpuErrchk(cudaMalloc(&d_r, Nnx_T));
    d_p = d_r;                      // share N*nx
    gpuErrchk(cudaMalloc(&d_v_temp, knotPoints * sizeof(T)));
    d_eta_new_temp = d_v_temp;      // share N

    /* Copy S, gamma, lambda from host to device */
    gpuErrchk(cudaMemcpy(d_S_in, h_S, 3 * Nnx2_T, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_lambda, h_lambda, Nnx_T, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_gamma, h_gamma, Nnx_T, cudaMemcpyHostToDevice));

    // construct the preconditioners and save to d_Pinv and d_H
    constructBlkTriDiagPrecondCooperativeKernel(stateSize, knotPoints,
                                                d_S_in, d_S_out, d_T, d_Pinv, d_H, d_gamma,
                                                config);

    uint32_t pcg_iters = solvePCGCooperativeKernel(stateSize, knotPoints,
                                                   d_S_out,
                                                   d_Pinv,
                                                   d_H,
                                                   d_gamma,
                                                   d_lambda,
                                                   d_r,
                                                   d_p,
                                                   d_v_temp,
                                                   d_eta_new_temp,
                                                   config);

    if (config->pcg_org_trans) {
        // TRANS
        // need to transform d_lambda back using d_T
        recover_lamdba(stateSize, knotPoints, d_T, d_lambda);
    }

    /* Copy data back */
    gpuErrchk(cudaMemcpy(h_lambda, d_lambda, Nnx_T, cudaMemcpyDeviceToHost));

    cudaFree(d_S_in);
    cudaFree(d_Pinv);
    cudaFree(d_lambda);
    cudaFree(d_gamma);
    if (config->pcg_poly_order > 0) {
        cudaFree(d_H);
    }
    cudaFree(d_r);
    cudaFree(d_v_temp);

    return pcg_iters;
}

/* constructBlkTriDiagPrecondCooperativeKernel is the interface function to be called from the device side (all data are declared within GPU, hence d_*)
 * it constructs the preconditioners d_Pinv and d_H
 * stateSize        -> integer, size of the block matrix. For OCP QP, this is nx.
 * knotPoints       -> integer, size of block rows. For OCP QP, this is N, the horizon length.
 * d_S              -> matrix, LHS of the linear equation, symmetric positive definite block tri-diagonal
 * d_T              -> matrix, if TRANS, it will be populated with transformations
 * d_Pinv           -> matrix, preconditioner, symmetric positive definite block tri-diagonal
 * d_H              -> matrix, preconditioner (if applicable), block tri-diagonal
 * d_gamma          -> vector, if TRANS, it will be transformed
 * config           -> configuration of the PCG algorithm
 */

template<typename T>
void constructBlkTriDiagPrecondCooperativeKernel(const uint32_t state_size,
                                                 const uint32_t knot_points,
                                                 T *d_S_in,
                                                 T *d_S_out,
                                                 T *d_T,
                                                 T *d_Pinv,
                                                 T *d_H,
                                                 T *d_gamma,
                                                 struct pcg_config<T> *config) {
    void *precondition_kernel = (void *) precondition<T, STATE_SIZE, KNOT_POINTS>;
    bool use_H = config->pcg_poly_order > 0;
    // the following shall be turned off for speed
    bool gpu_check = checkPreconditionOccupancy<T>(precondition_kernel, config->pcg_block, state_size, knot_points);
    // gpu_check shall always be true, o.w. the program exits
    // gpu_check true means
    //      1. Device supports Cooperative Threads
    //      2. Device has enough shared memory for the current state_size & knot_points

    void *kernelArgs[] = {
            (void *) &d_S_in,
            (void *) &d_S_out,
            (void *) &d_T,
            (void *) &d_Pinv,
            (void *) &d_H,
            (void *) &d_gamma,
            (void *) &config->pcg_org_trans,
            (void *) &config->chol_or_ldl,
            (void *) &use_H
    };
    size_t precondition_kernel_smem_size = preconditionSharedMemSize<T>(state_size);

    gpuErrchk(cudaLaunchCooperativeKernel(precondition_kernel, knot_points, pcg_constants::DEFAULT_BLOCK, kernelArgs,
                                          precondition_kernel_smem_size));
//    gpuErrchk(cudaPeekAtLastError());

    return;
}

/* solvePCG is the interface function to be called from the host side (all data are declared within CPU, hence h_*)
 * it solves the linear equation S * lambda = gamma using Preconditioned Conjugate Gradient (PCG) method
 * h_S          -> matrix, LHS of the linear equation, symmetric positive definite block tri-diagonal
 * h_Pinv       -> matrix, preconditioner, symmetric positive definite block tri-diagonal
 * h_H          -> matrix, preconditioner (if applicable), block tri-diagonal
 * h_gamma      -> vector, RHS of the linear equation
 * h_lambda     -> vector, placeholder for the initial guess and the final result
 * stateSize    -> integer, size of the block matrix. For OCP QP, this is nx.
 * knotPoints   -> integer, size of block rows. For OCP QP, this is N, the horizon length.
 * config       -> configuration of the PCG algorithm
 */

template<typename T>
uint32_t solvePCG(
        T *h_S,
        T *h_Pinv,
        T *h_H,
        T *h_gamma,
        T *h_lambda,
        unsigned stateSize,
        unsigned knotPoints,
        struct pcg_config<T> *config) {

    const uint32_t states_sq = stateSize * stateSize;
    const uint32_t Nnx_T = stateSize * knotPoints * sizeof(T);
    const uint32_t Nnx2_T = knotPoints * states_sq * sizeof(T);

    /* Create device memory d_S, d_Pinv,
     * d_gamma, d_lambda, d_r, d_p,
     * d_v_temp, d_eta_new_temp
     * d_H if applicable */

    T *d_S, *d_Pinv, *d_gamma, *d_lambda;
    if (config->pcg_org_trans) {
        gpuErrchk(cudaMalloc(&d_S, 2 * Nnx2_T + Nnx_T));
        gpuErrchk(cudaMalloc(&d_Pinv, 2 * Nnx2_T + Nnx_T));
    } else {
        gpuErrchk(cudaMalloc(&d_S, 3 * Nnx2_T));
        gpuErrchk(cudaMalloc(&d_Pinv, 3 * Nnx2_T));
    }
    gpuErrchk(cudaMalloc(&d_lambda, Nnx_T));
    gpuErrchk(cudaMalloc(&d_gamma, Nnx_T));

    T *d_H;
    if (config->pcg_poly_order > 0) {
        gpuErrchk(cudaMalloc(&d_H, 3 * Nnx2_T));
    }

    /*   PCG vars   */
    T *d_r, *d_p, *d_v_temp, *d_eta_new_temp;
    gpuErrchk(cudaMalloc(&d_r, Nnx_T));
    d_p = d_r;                      // share N*nx
    gpuErrchk(cudaMalloc(&d_v_temp, knotPoints * sizeof(T)));
    d_eta_new_temp = d_v_temp;      // share N


    /* Copy S, Pinv, gamma, lambda*/
    if (config->pcg_org_trans) {
        gpuErrchk(cudaMemcpy(d_S, h_S, 2 * Nnx2_T + Nnx_T, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_Pinv, h_Pinv, 2 * Nnx2_T + Nnx_T, cudaMemcpyHostToDevice));
    } else {
        gpuErrchk(cudaMemcpy(d_S, h_S, 3 * Nnx2_T, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_Pinv, h_Pinv, 3 * Nnx2_T, cudaMemcpyHostToDevice));
    }
    gpuErrchk(cudaMemcpy(d_lambda, h_lambda, Nnx_T, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_gamma, h_gamma, Nnx_T, cudaMemcpyHostToDevice));
    if (config->pcg_poly_order > 0) {
        gpuErrchk(cudaMemcpy(d_H, h_H, 3 * Nnx2_T, cudaMemcpyHostToDevice));
    }

    uint32_t pcg_iters = solvePCGCooperativeKernel(stateSize, knotPoints,
                                                   d_S,
                                                   d_Pinv,
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

    cudaFree(d_S);
    cudaFree(d_Pinv);
    cudaFree(d_lambda);
    cudaFree(d_gamma);
    if (config->pcg_poly_order > 0) {
        cudaFree(d_H);
    }
    cudaFree(d_r);
    cudaFree(d_v_temp);

    return pcg_iters;
}

/* solvePCGCooperativeKernel is the interface function to be called from the device side (all data are declared within GPU, hence d_*)
 * it solves the linear equation S * lambda = gamma using Preconditioned Conjugate Gradient (PCG) method
 * d_S              -> matrix, LHS of the linear equation, symmetric positive definite block tri-diagonal
 * d_Pinv           -> matrix, preconditioner, symmetric positive definite block tri-diagonal
 * d_H              -> matrix, preconditioner (if applicable), block tri-diagonal
 * d_gamma          -> vector, RHS of the linear equation 
 * d_lambda         -> vector, placeholder for the initial guess and the final result
 * d_r              -> vector, used for storing intermediate result
 * d_p              -> vector, used for storing intermediate result
 * d_v_temp         -> vector, used for storing intermediate result
 * d_eta_new_temp   -> vector, used for storing intermediate result
 * config           -> configuration of the PCG algorithm
 */

template<typename T>
uint32_t solvePCGCooperativeKernel(const uint32_t state_size,
                                   const uint32_t knot_points,
                                   T *d_S,
                                   T *d_Pinv,
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
    T *d_poly_coeff = NULL;
    if (config->pcg_poly_order > 0) {
        gpuErrchk(cudaMalloc(&d_poly_coeff, config->pcg_poly_order * sizeof(T)));
        gpuErrchk(cudaMemcpy(d_poly_coeff, config->pcg_poly_coeff, config->pcg_poly_order * sizeof(T),
                             cudaMemcpyHostToDevice));
    }


    void *pcg_kernel = (void *) pcg<T, STATE_SIZE, KNOT_POINTS>;

    // the following shall be turned off for speed
    bool gpu_check = checkPcgOccupancy<T>(pcg_kernel, config->pcg_block, state_size, knot_points, config->pcg_org_trans,
                                          config->pcg_poly_order);
    // gpu_check shall always be true, o.w. the program exits
    // gpu_check true means
    //      1. Device supports Cooperative Threads
    //      2. Device has enough shared memory for the current state_size & knot_points

    void *kernelArgs[] = {
            (void *) &d_S,
            (void *) &d_Pinv,
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
            (void *) &config->pcg_org_trans,
            (void *) &config->pcg_poly_order,
            (void *) &d_poly_coeff
    };
    uint32_t h_pcg_iters;

    size_t ppcg_kernel_smem_size = pcgSharedMemSize<T>(state_size, knot_points, config->pcg_org_trans,
                                                       config->pcg_poly_order);

   int blocksPerSM = 0;

    // Calcul du nombre total de threads par bloc
    int blockSize = pcg_constants::DEFAULT_BLOCK.x *
                  pcg_constants::DEFAULT_BLOCK.y *
                  pcg_constants::DEFAULT_BLOCK.z;

    // Appel correct
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocksPerSM,
      pcg_kernel,
      blockSize,
      ppcg_kernel_smem_size);

    if (err != cudaSuccess) {
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int numSM = prop.multiProcessorCount;

    printf("Number of SM: %d\n", numSM);
    printf("Activ block per SM : %d\n", blocksPerSM);
    printf("Shared memory per bloc : %.1f KB\n", ppcg_kernel_smem_size / 1024.0);
    printf("total memory size per SM : %.1f KB\n", (blocksPerSM * ppcg_kernel_smem_size) / 1024.0);
    printf("Grid dimensions  : (%d, %d, %d)\n", config->pcg_grid.x, config->pcg_grid.y, config->pcg_grid.z);
    printf("Block dimensions : (%d, %d, %d)\n", config->pcg_block.x, config->pcg_block.y, config->pcg_block.z);
    printf("totals threads   : %d\n", config->pcg_grid.x * config->pcg_grid.y * config->pcg_grid.z *
                                        config->pcg_block.x * config->pcg_block.y * config->pcg_block.z);



    gpuErrchk(cudaLaunchCooperativeKernel(pcg_kernel, knot_points, pcg_constants::DEFAULT_BLOCK, kernelArgs,
                                          ppcg_kernel_smem_size));


//    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMemcpy(&h_pcg_iters, d_pcg_iters, sizeof(uint32_t), cudaMemcpyDeviceToHost));


    gpuErrchk(cudaFree(d_pcg_iters));
    gpuErrchk(cudaFree(d_pcg_exit));
    if (config->pcg_poly_order > 0) {
        gpuErrchk(cudaFree(d_poly_coeff));
    }
    return h_pcg_iters;
}