#include <iostream>
#include <stdio.h>
#include "gpu_pcg.cuh"
#include "gpuassert.cuh"
#include "Gauss_Jordan.cuh"
#include "affichage.cuh"
#include "conjugate_gradient.cuh"
#include "test.cuh"

void testing_CG_no_GPU();
void test_2x2();
void test_pcg();


int main() {
  //testing_CG_no_GPU();
  //test_2x2();
  // test_pcg();
  test_matmul();
  

    return 0;
}

void testing_CG_no_GPU () {
  const int N = 10;

  float A[100] = {
      70,  20,  15,  14,  18,  19,  22,  13,  17,  21,
      20,  78,  16,  15,  14,  12,  18,  11,  19,  13,
      15,  16,  65,  20,  13,  14,  15,  17,  16,  18,
      14,  15,  20,  72,  19,  11,  12,  14,  15,  13,
      18,  14,  13,  19,  80,  20,  14,  15,  13,  12,
      19,  12,  14,  11,  20,  77,  18,  19,  16,  15,
      22,  18,  15,  12,  14,  18,  74,  20,  19,  17,
      13,  11,  17,  14,  15,  19,  20,  69,  18,  16,
      17,  19,  16,  15,  13,  16,  19,  18,  75,  14,
      21,  13,  18,  13,  12,  15,  17,  16,  14,  71
  };

  float b[N] = {
      1510, 1457, 1287, 1311, 1548, 1524, 1506, 1343, 1465, 1402
  };

  float x0[N] = {10, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // gaussJordan(A, b, x0, 10);
  Conjugate_Gradien(A, b, x0, N);
  printVector("x0", x0, N);

  float test[N];
  for (int i = 0; i < N; ++i) test[i] = 0;
  mat_mul_vector(A, x0, test, N);
  printVector("A @ x0", test, N);
  printVector("b", b, N);
}

void test_2x2() {
  //number example of wikipedia
  // https://en.wikipedia.org/wiki/Conjugate_gradient_method
  float* A = new float[4] {4, 1,
                          1, 3};
  float* b = new float[2] {1,2};
  float* x0 = new float[2] {2, 1};
  int size = 2;

  Conjugate_Gradien(A, b, x0, size);

  printVector("x0", x0, size);

  delete[] A;
  delete[] b;
  delete[] x0;
}

void test_data() {
  
}

void test_pcg() {
  float h_S[36] = {0,0,0,0,
                    -.999, 0, 0, -.999,
                    .999, .0999, -.98, .999,
                    .999, -.98, .0999, .999,
                    -2.008, .8801, .8801, -3.0584,
                    .999, .0999, -.98, .999,
                    .999, -.98, .0999, .999,
                    -1.019, .8801, .8801, -2.0694,
                    0,0,0,0};

    float h_Pinv[36];
    // initialize Pinv with some arbitrary values
    for (int i=0; i<36; i++){
        if (h_S[i] == 0){ h_Pinv[i] = 0; }
        else { h_Pinv[i] = 1 / h_S[i]; }
    }


    float h_gamma[6] = {3.1385, 0, 0, 3.0788, .0031, 3.0788};
    float h_lambda[6] = {0,0,0,0,0,0};
    float x[6] = {-303.706, -46.4152, -315.179, -14.8973, -298.792, 13.505};

    matVecProduct(h_S, x, h_lambda, 6);
    printVector("h_lambda", h_lambda, 6, 4);

    gaussJordan(h_S, h_gamma, h_lambda, 6);
    printVector("h_lambda", h_lambda, 6, 4);

    float A[4] = {1, 2, 
                3, 4};

    float b[2] = {5, 10};
    float y[2] = {0, 0};

    gaussJordan(A, b, y, 2);
    printVector("x", y, 2, 4);
}
