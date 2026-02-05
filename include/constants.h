#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#ifndef STATE_SIZE
#define STATE_SIZE 32
#endif

#ifndef KNOT_POINTS
#define KNOT_POINTS 32
#endif

#ifndef PCG_TYPE
#define PCG_TYPE false // false -> org, true -> trans
#endif

#ifndef PRECOND_POLY_ORDER
#define PRECOND_POLY_ORDER 1 // now supports poly_order = 0, 1, 2, bigger number will not make a huge difference
#endif

#ifndef CHOL_OR_LDL
#define CHOL_OR_LDL false
#endif

#ifndef VERBOSE
#define VERBOSE 0
#endif

#ifndef STATExKERNEL
#define STATExKERNEL 0
#endif

#ifndef KNOTxKERNEL
#define KNOTxKERNEL 0
#endif

#ifndef STATExCOMPUTER
#define STATExCOMPUTER 0
#endif

#ifndef KNOTxCOMPUTER
#define KNOTxCOMPUTER 0
#endif

#ifndef STATExNBR_ITERATION
#define STATExNBR_ITERATION 0
#endif

#ifndef KNOTxNBR_ITERATION
#define KNOTxNBR_ITERATION 0
#endif

#ifndef NBR_ITERATIONxERROR
#define NBR_ITERATIONxERROR 0
#endif

#ifndef KERNELxERROR
#define KERNELxERROR 1
#endif

#ifndef DOUBLE
#define DOUBLE 1
#endif

#ifndef OPTIMISED
#define OPTIMISED 1
#endif

#ifndef NBR_ITERATION_MAX
#define NBR_ITERATION_MAX 1000000
#endif

#ifndef DATA_PATH
#define DATA_PATH "../../include/data"
#endif

namespace pcg_constants
{
  uint32_t DEFAULT_MAX_PCG_ITER = 100000;
  template <typename T>
  T DEFAULT_EPSILON = 1e-7;
  dim3 DEFAULT_GRID(KNOT_POINTS); // one SMBlock per knot point
  #if not OPTIMISED
    dim3 DEFAULT_BLOCK(STATE_SIZE); // one thread per state variable
  #else
    dim3 DEFAULT_BLOCK(1024); // one warp per state variable
  #endif
  int sizeSM = 0;
  int sizeBlockShared = 0;
}

namespace gato {
namespace constants {
    constexpr uint32_t STATE_SIZE_SQ = STATE_SIZE * STATE_SIZE;
    constexpr uint32_t STATE_SQ_P_KNOTS = STATE_SIZE * STATE_SIZE * KNOT_POINTS; // Q, A
    constexpr uint32_t STATE_P_KNOTS = STATE_SIZE * KNOT_POINTS; // q, c
    constexpr uint32_t VEC_SIZE_PADDED = (KNOT_POINTS+2) * STATE_SIZE; // gamma
    constexpr uint32_t BLOCK_ROW_R_DIM = 3 * STATE_SIZE;
    constexpr uint32_t BLOCK_ROW_SIZE = 3 * STATE_SIZE * STATE_SIZE;
    constexpr uint32_t B3D_MATRIX_SIZE_PADDED = 3 * STATE_SIZE * STATE_SIZE * KNOT_POINTS; // S, P_inv
} // namespace constants
} // namespace gato

namespace sqp {

// -——————————————————compile time settings——————————————————

constexpr uint32_t PCG_THREADS = 1024;
constexpr uint32_t BATCH_SIZE = 1;

}  // namespace sqp