#pragma once

#include <stdint.h>
#include <cooperative_groups.h>
#include "types.cuh"
#include "glass.cuh"

namespace cgrps = cooperative_groups;

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

// The following shall be moved to MPCGPU include
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
