#pragma once

#include <stdint.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "types.cuh"
#include "gpuassert.cuh"
#include "utils.cuh"
#include "glass.cuh"

// https://github.com/NVIDIA/cuda-samples/blob/master/Samples/0_Introduction/simpleTemplates/sharedmem.cuh
#include "sharedmem.cuh"

namespace cgrps = cooperative_groups;

template<typename T>
size_t preconditionSharedMemSize(uint32_t state_size) {
    return sizeof(T) * (4 * state_size * state_size + 3 * state_size + 1);
}


template<typename T>
bool checkPreconditionOccupancy(void *kernel, dim3 block, uint32_t state_size, uint32_t knot_points) {

    const uint32_t smem_size = preconditionSharedMemSize<T>(state_size);
    // printf("[Precondition] shared memory per block in bytes = %d\n", smem_size);
    int dev = 0;

    int maxBytes = 65536; // this is 64 KB, corresponding to compute capability 7.5 (GTX 1650)
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxBytes);

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

/* this function assumes that d_S_in contains a dense s.p.d. blk. tri-diag. matrix
 * this function's task
 * if org_trans ->  1. fill in diagonal blocks of d_S_out
 *                  2. partially fill in off-diagonal blocks of d_S_out
 *                  2. fill in d_T
 *                  3. fill in the diagonal blocks of d_Pinv
 *                  4. transform d_gamma using d_T
 * if !org_trans -> 1. fill in the diagonal blocks of d_Pinv
 * check on May 13 -- OK.
 */

template<typename T>
__device__
void invert_transform_D_blockrow(uint32_t state_size, uint32_t knot_points,
                                 T *d_S_in, T *d_S_out, T *d_Pinv, T *d_T, T *d_gamma,
                                 T *s_temp, unsigned blockrow,
                                 bool org_trans, bool chol_or_ldl) {
    // the following is only useful for TRANS
    T *d_Sdb = d_S_out;
    T *d_Sob = d_S_out + knot_points * state_size;
    T *d_Pinvdb = d_Pinv;
    T *d_Pinvob = d_Pinv + knot_points * state_size;

    const uint32_t triangular_state = (1 + state_size) * state_size / 2;
    const uint32_t state_sq = state_size * state_size;

    // shared block memory usage: 4nx^2 + 3nx + 1

    T *s_M1 = s_temp;
    T *s_M2 = s_M1 + state_sq;          // 4 matrices of size nx^2
    T *s_M3 = s_M2 + state_sq;
    T *s_M4 = s_M3 + state_sq;
    T *s_v1 = s_M4 + state_sq;
    T *s_v2 = s_v1 + state_size;        // 3 vectors of size nx
    T *s_v3 = s_v2 + state_size + 1;    // +1 is very important due to matrix inversion needs 2nx+1 tmp
    T *s_end = s_v3 + state_size;

    if (blockrow == 0) {
        // load D1 to M1
        glass::copy<T>(state_sq, d_S_in + state_sq, s_M1);
        __syncthreads();

        if (org_trans) {
            // TRANS
            // do Cholesky or LDL' here on M1 = D1
            // note: D1 is dense but its lower triangular part is L1
            // note: v1 = \tilde{D}_k is diagonal
            if (chol_or_ldl) {
                // Cholesky, with square root
                glass::chol_InPlace<T>(state_size, s_M1);
                // v1 identity
                for (unsigned i = threadIdx.x; i < state_size; i += blockDim.x) {
                    s_v1[i] = static_cast<T>(1.0);
                }
            } else {
                // square root free Cholesky = LDL'
                glass::ldl_InPlace<T>(state_size, s_M1, s_v1);
            }
            // save v1 = \tilde{D}_1 to main diagonal S
            __syncthreads();
            store_block_db<T>(state_size, knot_points,
                              s_v1,
                              d_Sdb,
                              blockrow,
                              1
            );

            // invert the unit lower triangular L1
            // T1 = M2 <- inv(M1) = inv(L1)
            glass::loadIdentityTriangular<T>(state_size, s_M2);
            __syncthreads();
            glass::trsm_triangular<T, true>(state_size, s_M1, s_M2);

            // save M2 = T1 to d_T
            __syncthreads();
            glass::copy<T>(triangular_state, s_M2, d_T);

            // v1 <- inv(v1), calculate inv(\tilde{D}_1)
            for (unsigned i = threadIdx.x; i < state_size; i += blockDim.x) {
                s_v1[i] = static_cast<T>(1.0) / s_v1[i];
            }

            // save v1 = inv(\tilde{D}_1) to main diagonal Pinv
            __syncthreads();
            store_block_db<T>(state_size, knot_points,
                              s_v1,
                              d_Pinvdb,
                              blockrow,
                              1
            );

            // load gamma_1 to v2
            glass::copy<T>(state_size, d_gamma, s_v2);
            __syncthreads();

            // v3 <- M2 * v2 = T1 * gamma_1
            glass::trmv<T, false>(state_size, static_cast<T>(1.0), s_M2, s_v2, s_v3);

            // save v3 = T1 * gamma_1 in d_gamma
            __syncthreads();
            glass::copy<T>(state_size, s_v3, d_gamma);
        } else {
            // ORG
            // invert M1 = D1, M2 <- inv(M1) = D1_inv
            glass::loadIdentity<T>(state_size, s_M2);
            __syncthreads();
            glass::invertMatrix<T>(state_size, s_M1, s_v1);

            // save M2 = D1_inv to Pinv
            __syncthreads();
            store_block_bd<T>(state_size, knot_points,
                              s_M2,         // src
                              d_Pinv,       // dst
                              1,            // col
                              blockrow,     // blockrow
                              1             // positive
            );
        }
    } else {
        // load Dk to M1
        glass::copy<T>(state_sq, d_S_in + blockrow * 3 * state_sq + state_sq, s_M1);
        __syncthreads();
        if (org_trans) {
            // TRANS

            // do Cholesky or LDL' here on M1
            // note: M1 is dense but its lower triangular part is Lk
            // note: v1 = \tilde{D}_k is a diagonal matrix

            if (chol_or_ldl) {
                // Cholesky, with square root
                glass::chol_InPlace<T>(state_size, s_M1);
                // v1 identity
                for (unsigned i = threadIdx.x; i < state_size; i += blockDim.x) {
                    s_v1[i] = static_cast<T>(1.0);
                }
            } else {
                // square root free Cholesky = LDL'
                glass::ldl_InPlace<T>(state_size, s_M1, s_v1);
            }

            // save v1 = \tilde{D}_k to main diagonal S
            __syncthreads();
            store_block_db<T>(state_size, knot_points,
                              s_v1,
                              d_Sdb,
                              blockrow,
                              1
            );

            // invert the unit lower triangular Lk
            // M2 <- Tk = inv(Lk)
            glass::loadIdentityTriangular<T>(state_size, s_M2);
            __syncthreads();
            glass::trsm_triangular<T, true>(state_size, s_M1, s_M2);

            // save M2 = Tk to d_T
            __syncthreads();
            glass::copy<T>(triangular_state, s_M2, d_T + blockrow * triangular_state);

            // v1 <- inv(v1)
            for (unsigned i = threadIdx.x; i < state_size; i += blockDim.x) {
                s_v1[i] = static_cast<T>(1.0) / s_v1[i];
            }

            // save v1 = inv(\tilde{D}_k) to main diagonal Pinv
            __syncthreads();
            store_block_db<T>(state_size, knot_points,
                              s_v1,
                              d_Pinvdb,
                              blockrow,
                              1
            );

            // load gamma_k to v3
            glass::copy<T>(state_size, d_gamma + blockrow * state_size, s_v3);
            __syncthreads();

            // v1 <- M2 * v3 = Tk * gamma_k
            glass::trmv<T, false>(state_size, static_cast<T>(1.0), s_M2, s_v3, s_v1);

            // save v1 = Tk * gamma_k to d_gamma
            __syncthreads();
            glass::copy<T>(state_size, s_v1, d_gamma + blockrow * state_size);

            // load Okm1' to M1
            glass::copy<T>(state_sq, d_S_in + blockrow * state_sq * 3, s_M1);
            __syncthreads();

            // M3 <- M2 * M1 = Tk * phi_k = Tk * Okm1'
            glass::trmm_left<T, false>(state_size, state_size, static_cast<T>(1.0), s_M2, s_M1, s_M3);
            __syncthreads();

            // save M3 = Tk * Okm1' into left off-diagonal of S
            store_block_ob<T>(state_size, knot_points,
                              s_M3,         // src
                              d_Sob,        // dst
                              0,            // left block column
                              blockrow,     // blockrow
                              1             // positive
            );

            // load identity to M1
            glass::loadIdentity<T>(state_size, s_M1);
            __syncthreads();
            // M2 <- M1 * M3' = I * (Tk * Okm1')' = Okm1 * Tk'
            glass::gemm<T, true>(state_size, state_size, state_size, static_cast<T>(1.0), s_M1, s_M3, s_M2);

            // save M2 = Okm1 * Tk' into right off-diagonal of S
            __syncthreads();
            store_block_ob<T>(state_size, knot_points,
                              s_M2,         // src
                              d_Sob,        // dst
                              1,            // right block column
                              blockrow - 1, // blockrow
                              1             // positive
            );
        } else {
            // ORG
            glass::loadIdentity<T>(state_size, s_M2);
            // invert M1 = Dk, M2 <- inv(M1) = Dk_inv
            __syncthreads();
            glass::invertMatrix<T>(state_size, s_M1, s_v1);

            // save M2 = Dk_inv to Pinv
            __syncthreads();
            store_block_bd<T>(state_size, knot_points,
                              s_M2,         // src
                              d_Pinv,       // dst
                              1,            // middle block column
                              blockrow,     // blockrow
                              1             // positive
            );
        }
    }
}

/* this function's task
 * if org_trans ->  1. finish off-diagonal blocks of d_S_out
 *                  2. fill in off-diagonal blocks of d_Pinv
 *                  3. fill in diagonal blocks of d_H if needed
 * if !org_trans -> 1. fill in off-diagonal blocks of d_Pinv
 *                  2. complete H if needed
 * check on May 13 -- OK.
*/
template<typename T>
__device__
void complete_Pinv_blockrow(uint32_t state_size, uint32_t knot_points,
                            T *d_S_in, T *d_S_out, T *d_Pinv, T *d_H, T *d_T,
                            T *s_temp, unsigned blockrow,
                            bool org_trans, bool use_H) {

    const uint32_t states_sq = state_size * state_size;
    const unsigned lastrow = knot_points - 1;

    if (org_trans) {
        // TRANS
        const uint32_t triangular_state = (state_size + 1) * state_size / 2;

        T *d_Sob = d_S_out + knot_points * state_size;
        T *d_Pinvdb = d_Pinv;
        T *d_Pinvob = d_Pinv + knot_points * state_size;

        // shared block memory usage: 4nx^2 + 3nx

        T *s_M1 = s_temp;
        T *s_M2 = s_M1 + states_sq;
        T *s_M3 = s_M2 + states_sq;
        T *s_M4 = s_M3 + states_sq;
        T *s_v1 = s_M4 + states_sq;
        T *s_v2 = s_v1 + state_size;
        T *s_v3 = s_v2 + state_size;
        T *s_end = s_v3 + state_size;

        // load tilde_Dk_inv to v1
        load_block_db<T>(state_size, knot_points,
                         d_Pinvdb,          // src
                         s_v1,              // dst
                         blockrow           // blockrow
        );

        if (blockrow != lastrow) {
            // load Tk to M1. Note only the first triangular_state part of M1 is used.
            glass::copy<T>(triangular_state, d_T + blockrow * triangular_state, s_M1);

            // load Ok * Tkp1' to M2
            load_block_ob<T>(state_size, knot_points,
                             d_Sob,         // src
                             s_M2,          // dst
                             0,             // left block column
                             blockrow + 1,  // blockrow
                             true           // transpose
            );

            // M3 <- M1 * M2 = Tk * Ok * Tkp1' = tilde_Ok
            __syncthreads();
            glass::trmm_left<T, false>(state_size, state_size, static_cast<T>(1.0), s_M1, s_M2, s_M3);

            // save M3 = tilde_Ok to S (right diagonal)
            __syncthreads();
            store_block_ob<T>(state_size, knot_points,
                              s_M3,         // src
                              d_Sob,        // dst
                              1,            // right block column
                              blockrow,     // blockrow
                              1             // positive
            );

            // load tilde_Dkp1_inv to v2
            load_block_db<T>(state_size, knot_points,
                             d_Pinvdb,      // src
                             s_v2,          // dst
                             blockrow + 1   // blockrow
            );

            // M2 <- v1 * M3 (v1 is diagonal matrix) = tilde_Dk_inv * tilde_Ok
            glass::dimm_left<T>(state_size, state_size, static_cast<T>(1.0), s_v1, s_M3, s_M2);
            // M1 <- v2 * M2 (v2 is diagonal matrix) = tilde_Dk_inv * tilde_Ok * tilde_Dkp1_inv = tilde_Ek
            __syncthreads();
            glass::dimm_right<T>(state_size, state_size, static_cast<T>(1.0), s_v2, s_M2, s_M1);

            // save -M1 = -tilde_Ek to Pinv (right diagonal)
            __syncthreads();
            store_block_ob<T>(state_size, knot_points,
                              s_M1,         // src
                              d_Pinvob,     // dst
                              1,            // right block column
                              blockrow,     // blockrow
                              -1            // negative
            );

            if (use_H) {
                // M4 <- M1 * M3' = tilde_Ek * tilde_Ok'
                glass::gemm<T, true>(state_size, state_size, state_size, static_cast<T>(1.0), s_M1, s_M3, s_M4);
                __syncthreads();
                if (blockrow == 0) {
                    // save M4 = tilde_Ek * tilde_Ok' to H
                    store_block_bd<T>(state_size, knot_points,
                                      s_M4,     // src
                                      d_H,      // dst
                                      1,        // middle block column
                                      blockrow, // blockrow
                                      1         // positive
                    );
                }
            }
        }

        if (blockrow != 0) {
            // load Tkm1 to M1. Note only the first triangular_state part of M1 is used.
            glass::copy<T>(triangular_state, d_T + (blockrow - 1) * triangular_state, s_M1);

            // load Tk * Okm1' to M2
            load_block_ob<T>(state_size, knot_points,
                             d_Sob,         // src
                             s_M2,          // dst
                             0,             // left block column
                             blockrow       // block row
            );

            // M3 <- M2 * M1 = Tk * Okm1' * Tkm1' = tilde_Okm1'. Note trmm_right and transpose=true.
            __syncthreads();
            glass::trmm_right<T, true>(state_size, state_size, static_cast<T>(1.0), s_M1, s_M2, s_M3);

            // save M3 = tilde_Okm1' to S (left diagonal)
            __syncthreads();
            store_block_ob<T>(state_size, knot_points,
                              s_M3,         // src
                              d_Sob,        // dst
                              0,            // left block column
                              blockrow,     // blockrow
                              1             // positive
            );

            // load tilde_Dkm1_inv to v3
            load_block_db<T>(state_size, knot_points,
                             d_Pinvdb,      // src
                             s_v3,          // dst
                             blockrow - 1   // blockrow
            );

            // M2 <- v1 * M3 (v1 is diagonal matrix) = tilde_Dk_inv * tilde_Okm1'
            glass::dimm_left<T>(state_size, state_size, static_cast<T>(1.0), s_v1, s_M3, s_M2);
            // M1 <- v3 * M2 (v3 is diagonal matrix) = tilde_Dk_inv * tilde_Okm1' * tilde_Dkm1_inv = tilde_Ekm1'
            __syncthreads();
            glass::dimm_right<T>(state_size, state_size, static_cast<T>(1.0), s_v3, s_M2, s_M1);

            // save -M1 = -tilde_Ekm1' to Pinv (left diagonal)
            __syncthreads();
            store_block_ob<T>(state_size, knot_points,
                              s_M1,         // src
                              d_Pinvob,     // dst
                              0,            // left block column
                              blockrow,     // blockrow
                              -1            // negative
            );

            if (use_H) {
                // M2 <- M1 * M3' = tilde_Ekm1' * tilde_Okm1
                glass::gemm<T, true>(state_size, state_size, state_size, static_cast<T>(1.0), s_M1, s_M3, s_M2);
                if (blockrow != lastrow) {
                    // M2 <- M2 + M4 = Ekm1' * Okm1 + Ek * Ok'
                    __syncthreads();
                    for (unsigned ind = threadIdx.x; ind < states_sq; ind += blockDim.x) {
                        s_M2[ind] += s_M4[ind];
                    }
                }
                // save M2 = Ekm1' * Okm1 (lastrow) or Ekm1' * Okm1 + Ek * Ok' to H
                __syncthreads();
                store_block_bd<T>(state_size, knot_points,
                                  s_M2,     // src
                                  d_H,      // dst
                                  1,        // middle block column
                                  blockrow, // blockrow
                                  1         // positive
                );
            }
        }
    } else {
        // ORG
        // shared block memory usage: 4nx^2
        // could save one loading if 5nx^2 available

        T *s_M1 = s_temp;
        T *s_M2 = s_M1 + states_sq;
        T *s_M3 = s_M2 + states_sq;
        T *s_M4 = s_M3 + states_sq;
        T *s_end = s_M4 + states_sq;

        if (blockrow != lastrow) {
            // load Dk_inv to M1
            load_block_bd<T>(state_size, knot_points,
                             d_Pinv,        // src
                             s_M1,          // dst
                             1,             // middle block column
                             blockrow       // blockrow
            );

            // load Ok to M2
            load_block_bd<T>(state_size, knot_points,
                             d_S_in,        // src
                             s_M2,          // dst
                             0,             // left block column
                             blockrow + 1,  // block row
                             true           // transpose
            );

            // M3 <- M1 * M2 = Dk_inv * Ok
            __syncthreads();
            glass::gemm<T>(state_size, state_size, state_size, static_cast<T>(1.0), s_M1, s_M2, s_M3);

            // load Dkp1_inv to M1
            __syncthreads();
            load_block_bd<T>(state_size, knot_points,
                             d_Pinv,        // src
                             s_M1,          // dst
                             1,             // middle block column
                             blockrow + 1   // blockrow
            );

            // M4 = Ek <- M3 * M1 = Dk_inv * Ok * Dkp1_inv
            __syncthreads();
            glass::gemm<T>(state_size, state_size, state_size, static_cast<T>(1.0), s_M3, s_M1, s_M4);

            // save -M4 = -Ek to Pinv
            __syncthreads();
            store_block_bd<T>(state_size, knot_points,
                              s_M4,         // src
                              d_Pinv,       // dst
                              2,            // right block column
                              blockrow,     // blockrow
                              -1            // negative
            );

            if (use_H) {
                // M1 <- M4 * M2' = Ek * Ok'
                glass::gemm<T, true>(state_size, state_size, state_size, static_cast<T>(1.0), s_M4, s_M2, s_M1);

                // save M1 = Ek * Ok' to H, will be read again in the general case
                __syncthreads();
                store_block_bd<T>(state_size, knot_points,
                                  s_M1,     // src
                                  d_H,      // dst
                                  1,        // middle block column
                                  blockrow, // blockrow
                                  1         // positive
                );

                if (blockrow != lastrow - 1) {
                    // load Okp1 to M2
                    load_block_bd<T>(state_size, knot_points,
                                     d_S_in,        // src
                                     s_M2,          // dst
                                     0,             // left block column
                                     blockrow + 2,  // block row
                                     true           // transpose
                    );

                    // M3 <- M4 * M2 = Ek * Okp1
                    __syncthreads();
                    glass::gemm<T>(state_size, state_size, state_size, static_cast<T>(1.0), s_M4, s_M2, s_M3);

                    // save M3 = Ek * Okp1 to H
                    __syncthreads();
                    store_block_bd<T>(state_size, knot_points,
                                      s_M3,         // src
                                      d_H,          // dst
                                      2,            // right block column
                                      blockrow,     // blockrow
                                      1             // positive
                    );
                }
            }
        }

        if (blockrow != 0) {
            // load Dk_inv to M1
            load_block_bd<T>(state_size, knot_points,
                             d_Pinv,        // src
                             s_M1,          // dst
                             1,             // middle block column
                             blockrow       // blockrow
            );

            // load Okm1' to M2
            load_block_bd<T>(state_size, knot_points,
                             d_S_in,    // src
                             s_M2,      // dst
                             0,         // left block column
                             blockrow   // blockrow
            );

            // M3 <- M1 * M2 = Dk_inv * Okm1'
            __syncthreads();
            glass::gemm<T>(state_size, state_size, state_size, static_cast<T>(1.0), s_M1, s_M2, s_M3);

            // load Dkm1_inv to M1
            __syncthreads();
            load_block_bd<T>(state_size, knot_points,
                             d_Pinv,        // src
                             s_M1,          // dst
                             1,             // middle block column
                             blockrow - 1   // blockrow
            );

            // M4 = Ekm1' <- M3 * M1 = Dk_inv * Okm1' * Dkm1_inv
            __syncthreads();
            glass::gemm<T>(state_size, state_size, state_size, static_cast<T>(1.0), s_M3, s_M1, s_M4);

            // save -M4 = -Ekm1' to Pinv
            __syncthreads();
            store_block_bd<T>(state_size, knot_points,
                              s_M4,     // src
                              d_Pinv,   // dst
                              0,        // left block column
                              blockrow, // blockrow
                              -1        // negative
            );

            if (use_H) {
                // M1 <- M4 * M2' = Ekm1' * Okm1
                glass::gemm<T, true>(state_size, state_size, state_size, static_cast<T>(1.0), s_M4, s_M2, s_M1);

                if (blockrow != lastrow) {
                    // load M2 = Ek * Ok' (has been calculated before) from H
                    __syncthreads();
                    load_block_bd<T>(state_size, knot_points,
                                     d_H,       // src
                                     s_M2,      // dst
                                     1,         // middle block column
                                     blockrow   // blockrow
                    );
                    // M1 <- M1 + M2 = Ekm1' * Okm1 + Ek * Ok'
                    for (unsigned ind = threadIdx.x; ind < states_sq; ind += blockDim.x) {
                        s_M1[ind] += s_M2[ind];
                    }
                }

                // save M1 = Ekm1' * Okm1 (lastrow) or Ekm1' * Okm1 + Ek * Ok' to H
                __syncthreads();
                store_block_bd<T>(state_size, knot_points,
                                  s_M1,     // src
                                  d_H,      // dst
                                  1,        // middle block column
                                  blockrow, // blockrow
                                  1         // positive
                );

                if (blockrow != 1) {
                    // load phi_km1 = Okm2' to M2
                    load_block_bd<T>(state_size, knot_points,
                                     d_S_in,        // src
                                     s_M2,          // dst
                                     0,             // left block column
                                     blockrow - 1   // blockrow
                    );

                    // M3 <- M4 * M2 = Ekm1' * Okm2'
                    __syncthreads();
                    glass::gemm<T>(state_size, state_size, state_size, static_cast<T>(1.0), s_M4, s_M2, s_M3);

                    // save M3 = Ekm1' * Okm2' to H
                    __syncthreads();
                    store_block_bd<T>(state_size, knot_points,
                                      s_M3,     // src
                                      d_H,      // dst
                                      0,        // left block column
                                      blockrow, // blockrow
                                      1         // positive
                    );
                }
            }
        }
    }
}

// only for TRANS and use_H
// the diagonal blocks of d_H and the off-diagonal blocks of d_Pinv are ready
// compute the off-diagonal blocks for d_H, using d_Pinv and d_S
// check on May 13 -- OK.

template<typename T>
__device__
void complete_H_blockrow(uint32_t state_size, uint32_t knot_points,
                         T *d_S_out, T *d_Pinv, T *d_H,
                         T *s_temp, unsigned blockrow) {
    const unsigned lastrow = knot_points - 1;
    const uint32_t states_sq = state_size * state_size;

    T *d_Sob = d_S_out + knot_points * state_size;
    T *d_Pinvob = d_Pinv + knot_points * state_size;

    // shared block memory usage: 3nx^2

    T *s_M1 = s_temp;
    T *s_M2 = s_M1 + states_sq;
    T *s_M3 = s_M2 + states_sq;
    T *s_end = s_M3 + states_sq;

    if (blockrow > 1) {
        // not first two block rows

        // load -tilde_Ekm1' to M1
        load_block_ob<T>(state_size, knot_points,
                         d_Pinvob,      // src
                         s_M1,          // dst
                         0,             // left block column
                         blockrow       // blockrow
        );

        // load tilde_Okm2' to M2
        load_block_ob<T>(state_size, knot_points,
                         d_Sob,         // src
                         s_M2,          // dst
                         0,             // left block column
                         blockrow - 1   // blockrow
        );

        // M3 <- (-1) * -M1 * M2 = tilde_Ekm1' * tilde_Okm2'
        __syncthreads();
        glass::gemm<T>(state_size, state_size, state_size, static_cast<T>(-1.0), s_M1, s_M2, s_M3);

        // save M3 = tilde_Ekm1' * tilde_Okm2' to H
        __syncthreads();
        store_block_bd<T>(state_size, knot_points,
                          s_M3,     // src
                          d_H,      // dst
                          0,        // left block column
                          blockrow, // blockrow
                          1         // positive
        );
    }

    if (blockrow < lastrow - 1) {
        // not last two block rows

        // load -tilde_Ek to M1
        load_block_ob<T>(state_size, knot_points,
                         d_Pinvob,      // src
                         s_M1,          // dst
                         1,             // right block column
                         blockrow       // blockrow
        );

        // load tilde_Okp1 to M2
        load_block_ob<T>(state_size, knot_points,
                         d_Sob,         // src
                         s_M2,          // dst
                         1,             // right block column
                         blockrow + 1   // block row
        );

        // M3 <- (-1) * -M1 * M2 = tilde_Ek * tilde_Okp1
        __syncthreads();
        glass::gemm<T>(state_size, state_size, state_size, static_cast<T>(-1.0), s_M1, s_M2, s_M3);

        // save M3 = tilde_Ek * tilde_Okp1 to H
        __syncthreads();
        store_block_bd<T>(state_size, knot_points,
                          s_M3,         // src
                          d_H,          // dst
                          2,            // right block column
                          blockrow,     // blockrow
                          1             // positive
        );
    }
}


template<typename T, uint32_t state_size, uint32_t knot_points>
__global__
void precondition(
        T *d_S_in,     // always size = 3Nnx^2
        T *d_S_out, // if ORG, d_S_out = d_S_in; if TRANS, size = Nnx + 2Nnx^2, diagonal | off-diagonal blocks
        T *d_T,     // if ORG, d_T == NULL; if TRANS, size = N(nx + 1)nx/2
        T *d_Pinv,  // if ORG, size = 3Nnx^2; if TRANS, size = Nnx + 2Nnx^2, diagonal | off-diagonal blocks
        T *d_H,     // if poly_order == 0, d_H = NULL; if poly_order > 0, size = 3Nnx^2
        T *d_gamma, // if ORG, nothing happens; if TRANS, T*gamma
        bool org_trans,
        bool chol_or_ldl,
        bool use_H) {

    extern __shared__ T s_temp[];
    for (unsigned blockrow = blockIdx.x; blockrow < knot_points; blockrow += gridDim.x) {
        invert_transform_D_blockrow<T>(state_size, knot_points,
                                       d_S_in, d_S_out, d_Pinv, d_T, d_gamma,
                                       s_temp, blockrow, org_trans, chol_or_ldl);
    }
    cgrps::this_grid().sync();

    for (unsigned blockrow = blockIdx.x; blockrow < knot_points; blockrow += gridDim.x) {
        complete_Pinv_blockrow<T>(state_size, knot_points,
                                  d_S_in, d_S_out, d_Pinv, d_H, d_T,
                                  s_temp, blockrow, org_trans, use_H);
    }

    if (use_H && org_trans) {
        // TRANS and use_H, one more grid sync needed
        cgrps::this_grid().sync();
        for (unsigned blockrow = blockIdx.x; blockrow < knot_points; blockrow += gridDim.x) {
            complete_H_blockrow<T>(state_size, knot_points,
                                   d_S_out, d_Pinv, d_H,
                                   s_temp, blockrow);
        }
    }
}

// only needed when org_trans == true (TRANS)
template<typename T>
__global__
void recover_lambda_kernel(uint32_t state_size, uint32_t knot_points,
                           T *d_T, T *d_lambda) {

    extern __shared__ T s_mem[];
    const uint32_t triangular_state = (state_size + 1) * state_size / 2;

    for (unsigned blockrow = blockIdx.x; blockrow < knot_points; blockrow += gridDim.x) {

        // shared block memory usage: nx(nx+1)/2 + 2nx

        T *s_Tk = s_mem;
        T *s_lambda_k = s_Tk + triangular_state;
        T *s_lambdaNew_k = s_lambda_k + state_size;
        T *s_end = s_lambdaNew_k + state_size;

        glass::copy<T>(triangular_state, d_T + blockrow * triangular_state, s_Tk);
        glass::copy<T>(state_size, d_lambda + blockrow * state_size, s_lambda_k);

        // calculate Tk' * lambda_k
        __syncthreads();
        glass::trmv<T, true>(state_size, static_cast<T>(1.0), s_Tk, s_lambda_k, s_lambdaNew_k);

        // save s_lambdaNew_k to d_lambda
        glass::copy<T>(state_size, s_lambdaNew_k, d_lambda + blockrow * state_size);
    }
}