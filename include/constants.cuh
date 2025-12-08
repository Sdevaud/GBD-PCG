#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#ifndef STATE_SIZE
#define STATE_SIZE 3
#endif

#ifndef KNOT_POINTS
#define KNOT_POINTS 4
#endif

#ifndef PCG_TYPE
#define PCG_TYPE false // false -> org, true -> trans
#endif

#ifndef PRECOND_POLY_ORDER
#define PRECOND_POLY_ORDER false // now supports poly_order = 0, 1, 2, bigger number will not make a huge difference
#endif

#ifndef CHOL_OR_LDL
#define CHOL_OR_LDL false
#endif

#ifndef DEBUG
#define DEBUG false
#endif

#ifndef OPTIMISED
#define OPTIMISED false
#endif

#ifndef TIME_EXECUTION_DOUBLE
#define TIME_EXECUTION_DOUBLE false
#endif

#ifndef TIME_EXECUTION_FLOAT
#define TIME_EXECUTION_FLOAT false
#endif

#ifndef ERROR_DOUBLE
#define ERROR_DOUBLE false
#endif

#ifndef ERROR_FLOAT
#define ERROR_FLOAT false
#endif

#ifndef NBR_ITERATION
#define NBR_ITERATION 10000
#endif

namespace pcg_constants
{
  uint32_t DEFAULT_MAX_PCG_ITER = 10000;
  template <typename T>
  T DEFAULT_EPSILON = 1e-8;
  dim3 DEFAULT_GRID(11); // one SMBlock per knot point
  dim3 DEFAULT_BLOCK(1024); // one thread per state variable
  int sizeSM = 0;
  int sizeBlockShared = 0;
}
