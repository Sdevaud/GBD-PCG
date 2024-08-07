#pragma once

#include <stdint.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "types.cuh"
#include "gpuassert.cuh"
#include "utils.cuh"
#include "glass.cuh"

namespace cgrps = cooperative_groups;

template<typename T>
size_t pcgSharedMemSize(uint32_t state_size, uint32_t knot_points, int pcg_poly_order) {
    if (pcg_poly_order == 1) {
        // poly_order = 1, use I_H
        return sizeof(T) * (2 * 3 * state_size * state_size +       // dense S and Pinv
                            3 * state_size * state_size +       // H size
                            8 * state_size +
                            max(state_size, knot_points));
    } else if (pcg_poly_order == 0) {
        // poly_order = 0, don't use I_H
        return sizeof(T) * (2 * 3 * state_size * state_size +       // dense S and Pinv
                            6 * state_size +
                            max(state_size, knot_points));
    }

}


template<typename T>
bool checkPcgOccupancy(void *kernel, dim3 block, uint32_t state_size, uint32_t knot_points, int pcg_poly_order) {

    const uint32_t smem_size = pcgSharedMemSize<T>(state_size, knot_points, pcg_poly_order);
    int dev = 0;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    int supportsCoopLaunch = 0;
    gpuErrchk(cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev));
    if (!supportsCoopLaunch) {
        printf("[Error] Device does not support Cooperative Threads\n");
        exit(5);
    }

    int numProcs = deviceProp.multiProcessorCount;
    int numBlocksPerSm;
    gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel, block.x * block.y * block.z,
                                                            smem_size));

    if ((int) knot_points > numProcs * numBlocksPerSm) {
        printf("Too many knot points ([%d]). Device supports [%d] active blocks, over [%d] SMs.\n", knot_points,
               numProcs * numBlocksPerSm, numProcs);
        exit(6);
    }

    return true;
}


template<typename T, uint32_t state_size, uint32_t knot_points>
__global__
void pcg(
        T *d_S,
        T *d_Pinv,
        T *d_H,
        T *d_gamma,
        T *d_lambda,
        T *d_r,
        T *d_p,
        T *d_v_temp,
        T *d_eta_new_temp,
        uint32_t *d_iters,
        bool *d_max_iter_exit,
        uint32_t max_iter,
        T exit_tol,
        int poly_order) {

    const cgrps::thread_block block = cgrps::this_thread_block();
    const cgrps::grid_group grid = cgrps::this_grid();
    const uint32_t block_id = blockIdx.x;
    const uint32_t block_dim = blockDim.x;
    const uint32_t thread_id = threadIdx.x;
    const uint32_t block_x_statesize = block_id * state_size;
    const uint32_t states_sq = state_size * state_size;

    extern __shared__ T s_temp[];
    /* ---------- block shared mem declaration starts ----------*/

    T *s_S = s_temp;
    T *s_Pinv = s_S + 3 * states_sq;
    T *s_H, *s_v_b;
    if (poly_order == 1) {
        s_H = s_Pinv + 3 * states_sq;
        s_v_b = s_H + 3 * states_sq;        // H size = 3nx^2, if needed
    } else if (poly_order == 0) {
        s_v_b = s_Pinv + 3 * states_sq;
    }

    T *s_eta_new_b = s_v_b;                 // share max(N, nx)
    T *s_lambda_b = s_eta_new_b + max(knot_points, state_size);

    // TODO: A graph shall be included to explain the memory allocation
    T *s_lambda = s_lambda_b - state_size;

    // lambda_{b-1:b+1}, p_{b-1:b+1}, r_{b-1:b+1} all in consecutive mem. Important!!!
    T *s_end;                               // access beyond s_end is forbidden
    if (poly_order == 1) {
        s_end = s_lambda_b + 8 * state_size;
    } else if (poly_order == 0) {
        s_end = s_lambda_b + 6 * state_size;
    }
    T *s_p = s_lambda + 2 * state_size;
    T *s_r_tilde = s_p;
    T *s_r = s_lambda + 4 * state_size;

    T *s_p_b = s_p + state_size;
    T *s_r_b = s_r + state_size;
    T *s_gamma = s_r_b + state_size;
    T *s_upsilon = s_gamma;

    // r_extra is introduced for poly_order == 1 because after r_extra = Pinv * r_{b-1:b+1}
    // one extra step is needed: r_tilde = H * r_extra_{b-2, b, b+2}
    T *s_r_extra, *s_r_extra_b;
    if (poly_order == 1) {
        s_r_extra = s_lambda + 6 * state_size;
        s_r_extra_b = s_r_extra + state_size;
    }

    /* ---------- block shared mem declaration ends ----------*/

    /* ---------- block shared mem writing-in starts ----------*/

    // populate shared memory
    for (unsigned ind = thread_id; ind < 3 * states_sq; ind += block_dim) {
        if (block_id == 0 && ind < states_sq) { continue; }
        if (block_id == knot_points - 1 && ind >= 2 * states_sq) { continue; }

        s_S[ind] = d_S[block_id * states_sq * 3 + ind];
        s_Pinv[ind] = d_Pinv[block_id * states_sq * 3 + ind];
    }
    glass::copy<T>(state_size, &d_gamma[block_x_statesize], s_gamma);

    if (poly_order == 1) {
        // load H from GPU global mem to block shared mem
        for (unsigned ind = thread_id; ind < 3 * states_sq; ind += block_dim) {
            if ((block_id == 0 || block_id == 1) && ind < states_sq) { continue; }
            if ((block_id == knot_points - 1 || block_id == knot_points - 2) && ind >= 2 * states_sq) { continue; }

            s_H[ind] = d_H[block_id * states_sq * 3 + ind];
        }
    }

    /* ---------- block shared mem writing-in ends ----------*/

    uint32_t iter;
    T alpha, beta, eta, eta_new;
    T gamma_norm, r_norm;

    bool max_iter_exit = true;

    // compute norm of d_gamma (entire gamma), use s_eta_new_b & d_eta_new_temp temporarily
    __syncthreads();
    glass::dot<T, state_size>(s_eta_new_b, s_gamma, s_gamma);
    if (thread_id == 0) { d_eta_new_temp[block_id] = s_eta_new_b[0]; }
    grid.sync();
    glass::reduce<T>(s_eta_new_b, knot_points, d_eta_new_temp);
    __syncthreads();
    gamma_norm = pow(s_eta_new_b[0], 0.5);

    /* ---------- PCG starts (preparation phase) ----------*/

    // r = gamma - S * lambda
    loadVec_m1bp1<T, state_size, knot_points - 1>(s_lambda, block_id, &d_lambda[block_x_statesize]);
    __syncthreads();
    blk_tri_mv<T>(s_r_b, s_S, s_lambda, state_size, knot_points - 1, block_id);
    __syncthreads();
    for (unsigned ind = thread_id; ind < state_size; ind += block_dim) {
        s_r_b[ind] = s_gamma[ind] - s_r_b[ind];
        d_r[block_x_statesize + ind] = s_r_b[ind];
    }
    grid.sync();

    // r_tilde = Pinv * r
    loadVec_m1bp1<T, state_size, knot_points - 1>(s_r, block_id, &d_r[block_x_statesize]);
    __syncthreads();
    blk_tri_mv<T>(s_r_tilde, s_Pinv, s_r, state_size, knot_points - 1, block_id);
    __syncthreads();

    if (poly_order == 1) {
        for (uint32_t ind = thread_id; ind < state_size; ind += block_dim) {
            // save s_r_tilde to d_r globally
            d_r[block_x_statesize + ind] = s_r_tilde[ind];
            // load s_r_tilde to middle part of s_r_extra locally.
            s_r_extra[ind + state_size] = s_r_tilde[ind];
        }
        grid.sync();

        // r_tilde = H * r_extra
        // load first and last part of s_r_extra from d_r (global memory).
        loadVec_m2p2<T, state_size, knot_points - 1>(s_r_extra, block_id, &d_r[block_x_statesize]);
        __syncthreads();
        blk_penta_mv<T>(s_r_tilde, s_H, s_r_extra, state_size, knot_points - 1, block_id);
        __syncthreads();
    }

    // p = r_tilde
    for (unsigned ind = thread_id; ind < state_size; ind += block_dim) {
        s_p_b[ind] = s_r_tilde[ind];
        d_p[block_x_statesize + ind] = s_p_b[ind];
    }

    // eta = r * r_tilde
    glass::dot<T, state_size>(s_eta_new_b, s_r_b, s_r_tilde);
    if (thread_id == 0) { d_eta_new_temp[block_id] = s_eta_new_b[0]; }
    grid.sync();
    glass::reduce<T>(s_eta_new_b, knot_points, d_eta_new_temp);
    __syncthreads();
    eta = s_eta_new_b[0];

    /* ---------- PCG MAIN LOOP starts ----------*/

    for (iter = 0; iter < max_iter; iter++) {
        // upsilon = S * p
        loadVec_m1p1<T, state_size, knot_points - 1>(s_p, block_id, &d_p[block_x_statesize]);
        __syncthreads();
        blk_tri_mv<T>(s_upsilon, s_S, s_p, state_size, knot_points - 1, block_id);
        __syncthreads();

        // alpha = eta / p * upsilon
        glass::dot<T, state_size>(s_v_b, s_p_b, s_upsilon);
        __syncthreads();
        if (thread_id == 0) { d_v_temp[block_id] = s_v_b[0]; }
        grid.sync();
        glass::reduce<T>(s_v_b, knot_points, d_v_temp);
        __syncthreads();
        alpha = eta / s_v_b[0];

        // lambda = lambda + alpha * p
        // r = r - alpha * upsilon
        for (uint32_t ind = thread_id; ind < state_size; ind += block_dim) {
            s_lambda_b[ind] += alpha * s_p_b[ind];
            s_r_b[ind] -= alpha * s_upsilon[ind];
            d_r[block_x_statesize + ind] = s_r_b[ind];
        }
        grid.sync();

        // r_tilde = Pinv * r
        loadVec_m1p1<T, state_size, knot_points - 1>(s_r, block_id, &d_r[block_x_statesize]);
        __syncthreads();
        blk_tri_mv<T>(s_r_tilde, s_Pinv, s_r, state_size, knot_points - 1, block_id);
        __syncthreads();

        if (poly_order == 1) {
            for (uint32_t ind = thread_id; ind < state_size; ind += block_dim) {
                // save s_r_tilde to d_r globally
                d_r[block_x_statesize + ind] = s_r_tilde[ind];
                // load s_r_tilde to middle part of s_r_extra locally.
                s_r_extra[ind + state_size] = s_r_tilde[ind];
            }
            grid.sync();

            // r_tilde = H * r_extra
            // load first and last part of s_r_extra from d_r (global memory).
            loadVec_m2p2<T, state_size, knot_points - 1>(s_r_extra, block_id, &d_r[block_x_statesize]);
            __syncthreads();
            blk_penta_mv<T>(s_r_tilde, s_H, s_r_extra, state_size, knot_points - 1, block_id);
            __syncthreads();
        }

        // eta = r * r_tilde
        glass::dot<T, state_size>(s_eta_new_b, s_r_b, s_r_tilde);
        __syncthreads();
        if (thread_id == 0) { d_eta_new_temp[block_id] = s_eta_new_b[0]; }
        grid.sync();
        glass::reduce<T>(s_eta_new_b, knot_points, d_eta_new_temp);
        __syncthreads();
        eta_new = s_eta_new_b[0];

        // compute norm of r
        glass::dot<T, state_size>(s_eta_new_b, s_r_b, s_r_b);
        __syncthreads();
        if (thread_id == 0) { d_eta_new_temp[block_id] = s_eta_new_b[0]; }
        grid.sync();
        glass::reduce<T>(s_eta_new_b, knot_points, d_eta_new_temp);
        __syncthreads();
        r_norm = pow(s_eta_new_b[0], 0.5);

        // check exit condition
        if (r_norm / gamma_norm < exit_tol) {
            iter++;
            max_iter_exit = false;
            break;
        }

        // beta = eta_new / eta
        // eta = eta_new
        beta = eta_new / eta;
        eta = eta_new;

        // p = r_tilde + beta*p
        for (uint32_t ind = thread_id; ind < state_size; ind += block_dim) {
            s_p_b[ind] = s_r_tilde[ind] + beta * s_p_b[ind];
            d_p[block_x_statesize + ind] = s_p_b[ind];
        }
        grid.sync();
    }
    /* ---------- PCG ends ----------*/

    // save output
    if (block_id == 0 && thread_id == 0) {
        d_iters[0] = iter;
        d_max_iter_exit[0] = max_iter_exit;
    }

    glass::copy<T>(state_size, s_lambda_b, &d_lambda[block_x_statesize]);

    grid.sync();
}