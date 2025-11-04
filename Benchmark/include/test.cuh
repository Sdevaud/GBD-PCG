#include <stdio.h>
#include <stdlib.h>
#include <cmath> 
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <iostream>
#include <vector>
#include "generate_A_SPD.cuh"

template<typename T>
T* create_matrix_test(const int state_size = 5, const int knot_points = 5) {
  T* h_S = generate_spd_block_tridiagonal(state_size, knot_points);
  T* h_gamma = generate_random_vector(Nnx);

  return A;
}
