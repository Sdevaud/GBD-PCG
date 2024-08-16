#pragma once

#include <cstdint>              // for uint32_t
#include <cuda_runtime.h>       // for dim3
#include "constants.cuh"

template<typename T>
struct pcg_config {

    T pcg_exit_tol;
    uint32_t pcg_max_iter;

    dim3 pcg_grid;
    dim3 pcg_block;

    bool pcg_org_trans; // false -> org, true -> trans

    int pcg_poly_order; // now supports poly_order = 0, 1, 2
    T pcg_poly_coeff[PRECOND_POLY_ORDER];

    pcg_config(T exit_tol = pcg_constants::DEFAULT_EPSILON<T>,
               uint32_t max_iter = pcg_constants::DEFAULT_MAX_PCG_ITER,
               dim3 grid = pcg_constants::DEFAULT_GRID,
               dim3 block = pcg_constants::DEFAULT_BLOCK,
               bool org_trans = PCG_TYPE,
               int poly_order = PRECOND_POLY_ORDER)
            :
            pcg_exit_tol(exit_tol), pcg_max_iter(max_iter), pcg_grid(grid), pcg_block(block),
            pcg_org_trans(org_trans), pcg_poly_order(poly_order) {}
};