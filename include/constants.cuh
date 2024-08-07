#pragma once
#include <cstdint>
#include <cuda_runtime.h>

#ifndef STATE_SIZE
#define STATE_SIZE  3
#endif

#ifndef KNOT_POINTS
#define KNOT_POINTS  3
#endif

#ifndef PRECOND_POLY_ORDER
#define PRECOND_POLY_ORDER  1  // now supports poly_order = 0, 1, future may support = 2
#endif

namespace pcg_constants{
    int DEFAULT_PRECOND_POLY_ORDER = 1;
    uint32_t DEFAULT_MAX_PCG_ITER = 1000;
	template<typename T>
    T DEFAULT_EPSILON = 1e-8;
    dim3 DEFAULT_GRID(128);
    dim3 DEFAULT_BLOCK(32);     // should be >= 32 because one warp contains 32 threads
}
