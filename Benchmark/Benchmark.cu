#include <iostream>
#include <stdio.h>
#include "gpu_pcg.cuh"
#include "gpuassert.cuh"
#include "Gauss_Jordan.cuh"
#include "affichage.cuh"
#include "conjugate_gradient.cuh"
#include "test.cuh"


int main() {
  //testing_CG_no_GPU();
  //test_2x2();
  // test_pcg();
  test_matmul();
  

    return 0;
}


