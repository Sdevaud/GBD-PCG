#pragma once

#include <stdint.h>
#include <cooperative_groups.h>
#include "types.cuh"
#include "glass.cuh"

namespace cgrps = cooperative_groups;

template<typename T, uint32_t block_dim, uint32_t max_block_id>
__device__
void loadVec_m1bp1(T *s_var,
                   const uint32_t block_id,
                   T *d_var_b) {

    // Need to load the middle part, m1bp1 means load b-1, b, b+1

    for (unsigned ind = threadIdx.x; ind < block_dim; ind += blockDim.x) {
        s_var[ind + block_dim] = *(d_var_b + ind);
    }

    if (block_id == 0) {
        for (unsigned ind = threadIdx.x; ind < block_dim; ind += blockDim.x) {
            s_var[ind + 2 * block_dim] = *(d_var_b + block_dim + ind);
        }
    } else if (block_id == max_block_id) {
        for (unsigned ind = threadIdx.x; ind < block_dim; ind += blockDim.x) {
            s_var[ind] = *(d_var_b - block_dim + ind);
        }
    } else {
        T *dst, *src;
        for (unsigned ind = threadIdx.x; ind < 2 * block_dim; ind += blockDim.x) {
            dst = s_var + ind + (ind >= block_dim) * block_dim;
            src = d_var_b + ind - (ind < block_dim) * block_dim;
            *dst = *src;
        }
    }
}


template<typename T, uint32_t block_dim, uint32_t max_block_id>
__device__
void loadVec_m1p1(T *s_var,
                  const uint32_t block_id,
                  T *d_var_b) {

    // no need to load the middle part, m1p1 means load b-1 and b+1

    if (block_id == 0) {
        for (unsigned ind = threadIdx.x; ind < block_dim; ind += blockDim.x) {
            s_var[ind + 2 * block_dim] = *(d_var_b + block_dim + ind);
        }
    } else if (block_id == max_block_id) {
        for (unsigned ind = threadIdx.x; ind < block_dim; ind += blockDim.x) {
            s_var[ind] = *(d_var_b - block_dim + ind);
        }
    } else {
        T *dst, *src;
        for (unsigned ind = threadIdx.x; ind < 2 * block_dim; ind += blockDim.x) {
            dst = s_var + ind + (ind >= block_dim) * block_dim;
            src = d_var_b + ind - (ind < block_dim) * block_dim;
            *dst = *src;
        }
    }
}

template<typename T, uint32_t block_dim, uint32_t max_block_id>
__device__
void loadVec_m2p2(T *s_var,
                  const uint32_t block_id,
                  T *d_var_b) {

    // no need to load the middle part, m2p2 means load b-2 and b+2

    if (block_id == 0 || block_id == 1) {
        for (unsigned ind = threadIdx.x; ind < block_dim; ind += blockDim.x) {
            s_var[ind + 2 * block_dim] = *(d_var_b + 2 * block_dim + ind);
        }
    } else if (block_id == max_block_id || block_id == max_block_id - 1) {
        for (unsigned ind = threadIdx.x; ind < block_dim; ind += blockDim.x) {
            s_var[ind] = *(d_var_b - 2 * block_dim + ind);
        }
    } else {
        T *dst, *src;
        for (unsigned ind = threadIdx.x; ind < 2 * block_dim; ind += blockDim.x) {
            dst = s_var + ind + (ind >= block_dim) * block_dim;
            if (ind < block_dim) {
                src = d_var_b + ind - 2 * block_dim;
            } else {
                src = d_var_b + ind + block_dim;
            }
            *dst = *src;
        }
    }
}

// for pcg.cuh, dense block tri-diagonal matrix vector multiplication

template<typename T>
__device__
void blk_tri_mv(T *s_dst,
                T *s_mat,
                T *s_vec,
                uint32_t b_dim,
                uint32_t max_block_id,
                uint32_t block_id) {
    // s_vec contains the 3 vectors: b-1, b, b+1
    // s_mat contains the 3 matrix blocks: left, middle, right
    // s_dst = s_mat * {b-1, b, b+1}

    T val;

    if (block_id == 0) {
        for (unsigned r = threadIdx.x; r < b_dim; r += blockDim.x) {
            val = static_cast<T>(0);
            for (unsigned c = 0; c < 2 * b_dim; c++) {
                val += s_mat[b_dim * b_dim + b_dim * c + r] * s_vec[c + b_dim]; // var and var+1
            }
            s_dst[r] = val;
        }
    } else if (block_id == max_block_id) {
        for (unsigned r = threadIdx.x; r < b_dim; r += blockDim.x) {
            val = static_cast<T>(0);
            for (unsigned c = 0; c < 2 * b_dim; c++) {
                val += s_mat[b_dim * c + r] * s_vec[c];
            }
            s_dst[r] = val;
        }
    } else {
        for (unsigned r = threadIdx.x; r < b_dim; r += blockDim.x) {
            val = static_cast<T>(0);
            for (unsigned c = 0; c < 3 * b_dim; c++) {
                val += s_mat[b_dim * c + r] * s_vec[c];
            }
            s_dst[r] = val;
        }
    }
}



// for pcg_trans.cuh, if I_H is used, poly_order = 1, sparse block penta-diagonal matrix vector multiplication

template<typename T>
__device__
void blk_penta_mv(T *s_dst,
                  T *s_mat,
                  T *s_vec,
                  uint32_t b_dim,
                  uint32_t max_block_id,
                  uint32_t block_id) {

    T val;

    if (block_id == 0 || block_id == 1) {
        for (unsigned r = threadIdx.x; r < b_dim; r += blockDim.x) {
            val = static_cast<T>(0);
            for (unsigned c = 0; c < 2 * b_dim; c++) {
                val += s_mat[b_dim * b_dim + b_dim * c + r] * s_vec[c + b_dim]; // var and var+1
            }
            s_dst[r] = val;
        }
    } else if (block_id == max_block_id || block_id == max_block_id - 1) {
        for (unsigned r = threadIdx.x; r < b_dim; r += blockDim.x) {
            val = static_cast<T>(0);
            for (unsigned c = 0; c < 2 * b_dim; c++) {
                val += s_mat[b_dim * c + r] * s_vec[c];
            }
            s_dst[r] = val;
        }
    } else {
        for (unsigned r = threadIdx.x; r < b_dim; r += blockDim.x) {
            val = static_cast<T>(0);
            for (unsigned c = 0; c < 3 * b_dim; c++) {
                val += s_mat[b_dim * c + r] * s_vec[c];
            }
            s_dst[r] = val;
        }
    }
}

// for pcg_trans.cuh, sparse block tri-diagonal matrix vector multiplication

template<typename T>
__device__
void blk_tri_mv_spa(T *s_dst,   // size = b_dim
                    T *s_matdb, // diagonal matrix, size = b_dim
                    T *s_matob, // left & right block matrices, size = 2 * b_dim * b_dim
                    T *s_vec,   // size = 3 * b_dim
                    uint32_t b_dim,
                    uint32_t max_block_id,
                    uint32_t block_id) {
    // s_vec contains three vectors: b-1, b, b+1
    // s_matdb contains the middle matrix block (diagonal)
    // s_matob contains the left and right matrix blocks (dense)
    // s_dst = s_matdb * b + s_matob * {b-1, b+1}

    T val;

    if (block_id == 0) {
        for (unsigned r = threadIdx.x; r < b_dim; r += blockDim.x) {
            val = static_cast<T>(0);
            for (unsigned c = 0; c < b_dim; c++) {
                // only right off-diagonal block times 3rd part of vector
                val += s_matob[b_dim * b_dim + b_dim * c + r] * s_vec[c + 2 * b_dim]; // var and var+1
            }
            // diagonal block times 2nd part of vector
            val += s_matdb[r] * s_vec[b_dim + r];
            s_dst[r] = val;
        }
    } else if (block_id == max_block_id) {
        for (unsigned r = threadIdx.x; r < b_dim; r += blockDim.x) {
            val = static_cast<T>(0);
            for (unsigned c = 0; c < b_dim; c++) {
                // only left off-diagonal block times 1st part of vector
                val += s_matob[b_dim * c + r] * s_vec[c];
            }
            // diagonal block times 2nd part of vector
            val += s_matdb[r] * s_vec[b_dim + r];
            s_dst[r] = val;
        }
    } else {
        for (unsigned r = threadIdx.x; r < b_dim; r += blockDim.x) {
            val = static_cast<T>(0);
            for (unsigned c = 0; c < b_dim; c++) {
                // left off-diagonal block times 1st part of vector
                val += s_matob[b_dim * c + r] * s_vec[c];
                // right off-diagonal block times 3rd part of vector
                val += s_matob[b_dim * b_dim + b_dim * c + r] * s_vec[c + 2 * b_dim];
            }
            // diagonal block times 2nd part of vector
            val += s_matdb[r] * s_vec[b_dim + r];
            s_dst[r] = val;
        }
    }
}

// r_tilde = (I + a*H + b*H^2) * r_tilde
// only for poly_order > 0
template<typename T, uint32_t b_dim, uint32_t max_block_id>
__device__
void I_H_mv(T *s_r_tilde,
            T *s_r_extra,
            T *s_v_b,
            T *s_H,
            T *d_r,
            T *poly_coeff,
            cgrps::grid_group grid,
            int poly_order,
            uint32_t block_id) {

    uint32_t block_x_statesize = block_id * b_dim;
    T scalar;

    // assume poly_order is at least 1
    int index = poly_order;
    while (index > 0) {
        for (uint32_t ind = threadIdx.x; ind < b_dim; ind += blockDim.x) {
            // save s_r_tilde to d_r globally
            d_r[block_x_statesize + ind] = s_r_tilde[ind];
            // load s_r_tilde to middle part of s_r_extra locally.
            s_r_extra[ind + b_dim] = s_r_tilde[ind];

            if (index < poly_order) {
                s_r_tilde[ind] = s_r_tilde[ind] * scalar + s_v_b[ind];
            }

            // store s_r_tilde to s_v_b temporarily for later addition
            s_v_b[ind] = s_r_tilde[ind];
        }
        grid.sync();

        // r_tilde = H * r_extra
        // load first and last part of s_r_extra from d_r (global memory).
        loadVec_m2p2<T, b_dim, max_block_id>(s_r_extra, block_id, &d_r[block_x_statesize]);
        __syncthreads();
        blk_penta_mv<T>(s_r_tilde, s_H, s_r_extra, b_dim, max_block_id, block_id);
        __syncthreads();
        scalar = poly_coeff[poly_order - index];

        index -= 1;
    }

    for (uint32_t ind = threadIdx.x; ind < b_dim; ind += blockDim.x) {
        // poly_order == 1   -> r_tilde = H * r_tilde,  v_b = r_tilde, scalar = a
        // poly_order == 2   -> r_tilde = H^2 * r_tilde,  v_b = (I + a*H) * r_tilde, scalar = b
        // poly_order == 3   -> r_tilde = H^3 * r_tilde,  v_b = (I + a*H + b*H^2) * r_tilde, scalar = c
        // ...
        s_r_tilde[ind] = s_r_tilde[ind] * scalar + s_v_b[ind];
    }
    __syncthreads();
}

// The following shall be moved to MPCGPU include
template<typename T>
__device__
void gato_memcpy(T *dst, T *src, unsigned size_Ts) {
    unsigned ind;
    for (ind = threadIdx.x; ind < size_Ts; ind += blockDim.x) {
        dst[ind] = src[ind];
    }
}

template<typename T>
__device__
void load_block_bd(uint32_t b_dim, uint32_t m_dim, T *src, T *dst, unsigned bcol, unsigned brow, bool transpose = false,
                   cooperative_groups::thread_group g = cooperative_groups::this_thread_block()) {

    if (bcol > 2 || brow > m_dim - 1) {
        printf("doing somehting wrong in load_block_bd\n");
        return;
    }


    unsigned block_row_offset, block_col_offset;

    block_row_offset = brow * (3 * b_dim * b_dim);
    block_col_offset = bcol * b_dim * b_dim;

    if (!transpose) {

        gato_memcpy<T>(
                dst,
                src + block_row_offset + block_col_offset,
                b_dim * b_dim
        );

    } else {

        unsigned ind, transpose_col, transpose_row;

        for (ind = threadIdx.x; ind < b_dim * b_dim; ind += blockDim.x) {
            transpose_col = ind % b_dim * b_dim;
            transpose_row = ind / b_dim;
            dst[transpose_col + transpose_row] = src[block_row_offset + block_col_offset + ind];
        }
    }
}

template<typename T>
__device__
void store_block_bd(uint32_t b_dim, uint32_t m_dim, T *src, T *dst, unsigned col, unsigned BLOCKNO, int multiplier = 1,
                    cooperative_groups::thread_group g = cooperative_groups::this_thread_block()) {

    unsigned block_row_offset, block_col_offset, ind;


    block_row_offset = BLOCKNO * (3 * b_dim * b_dim);
    block_col_offset = col * b_dim * b_dim;


    if (multiplier == 1) {

        glass::copy<T>(b_dim * b_dim, src, &dst[block_row_offset + block_col_offset]);

        gato_memcpy<T>(
                dst + block_row_offset + block_col_offset,
                src,
                b_dim * b_dim
        );

    } else {

        for (ind = g.thread_rank(); ind < b_dim * b_dim; ind += g.size()) {
            dst[block_row_offset + block_col_offset + ind] = src[ind] * multiplier;
        }

    }
}

// by Shaohui Yang July 16
template<typename T>
__device__
void store_block_db(uint32_t b_dim, uint32_t m_dim, T *src, T *dst, unsigned BLOCKNO, int multiplier = 1,
                    cooperative_groups::thread_group g = cooperative_groups::this_thread_block()) {

    unsigned block_row_offset, ind;
    block_row_offset = BLOCKNO * b_dim;

    if (multiplier == 1) {
        gato_memcpy<T>(
                dst + block_row_offset,
                src,
                b_dim
        );
    } else {
        for (ind = g.thread_rank(); ind < b_dim; ind += g.size()) {
            dst[block_row_offset + ind] = src[ind] * multiplier;
        }
    }
}

template<typename T>
__device__
void load_block_db(uint32_t b_dim, uint32_t m_dim, T *src, T *dst, unsigned brow,
                   cooperative_groups::thread_group g = cooperative_groups::this_thread_block()) {
    unsigned block_row_offset;
    block_row_offset = brow * b_dim;

    gato_memcpy<T>(
            dst,
            src + block_row_offset,
            b_dim
    );
}

template<typename T>
__device__
void load_block_ob(uint32_t b_dim, uint32_t m_dim, T *src, T *dst, unsigned bcol, unsigned brow, bool transpose = false,
                   cooperative_groups::thread_group g = cooperative_groups::this_thread_block()) {
    unsigned block_row_offset, block_col_offset;

    block_row_offset = brow * (2 * b_dim * b_dim);
    block_col_offset = bcol * b_dim * b_dim;

    if (!transpose) {
        gato_memcpy<T>(
                dst,
                src + block_row_offset + block_col_offset,
                b_dim * b_dim
        );
    } else {
        unsigned ind, transpose_col, transpose_row;
        for (ind = threadIdx.x; ind < b_dim * b_dim; ind += blockDim.x) {
            transpose_col = ind % b_dim * b_dim;
            transpose_row = ind / b_dim;
            dst[transpose_col + transpose_row] = src[block_row_offset + block_col_offset + ind];
        }
    }
}

template<typename T>
__device__
void store_block_ob(uint32_t b_dim, uint32_t m_dim, T *src, T *dst, unsigned col, unsigned BLOCKNO, int multiplier = 1,
                    cooperative_groups::thread_group g = cooperative_groups::this_thread_block()) {

    unsigned block_row_offset, block_col_offset, ind;

    block_row_offset = BLOCKNO * (2 * b_dim * b_dim);
    block_col_offset = col * b_dim * b_dim;

    if (multiplier == 1) {
        gato_memcpy<T>(
                dst + block_row_offset + block_col_offset,
                src,
                b_dim * b_dim
        );
    } else {
        for (ind = g.thread_rank(); ind < b_dim * b_dim; ind += g.size()) {
            dst[block_row_offset + block_col_offset + ind] = src[ind] * multiplier;
        }
    }
}