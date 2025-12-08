#include <iostream>
#include <cooperative_groups.h>
#include <iomanip>
#include "generate_A_SPD.cuh"
#include "CG_no_GPU.cuh"
#include "cu_solver.cuh"


template<typename T>
bool isSymmetric(const T* A, int N, T tol = 1e-12)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = i+1; j < N; j++)
        {
            T aij = A[i*N + j];
            T aji = A[j*N + i];

            if (std::fabs(aij - aji) > tol)
            {
                std::cout << "Not symmetric at (" << i << "," << j << ") : "
                          << aij << " != " << aji << std::endl;
                return false;
            }
        }
    }
    return true;
}

// =============================================================
// 1) Compter les non-zéros du triangle inférieur
// =============================================================
template<typename T>
int countLowerTriangleNNZ(const T* A, int N, T tol = 0)
{
    int nnz = 0;
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j <= i; j++)
        {
            if (std::fabs(A[i*N + j]) > tol)
                nnz++;
        }
    }
    return nnz;
}

// =============================================================
// 2) Construit rowptr (taille N+1)
// =============================================================
template<typename T>
void buildCSR_RowPtr(const T* A, int N, int* rowptr, T tol = 0)
{
    rowptr[0] = 0;
    int count = 0;

    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j <= i; j++)      // triangle inférieur
        {
            if (std::fabs(A[i*N + j]) > tol)
                count++;
        }
        rowptr[i+1] = count;
    }
}

// =============================================================
// 3) Construit colind[] et values[] (taille nnz)
// =============================================================
template<typename T>
void buildCSR_ColVal(const T* A, int N,
                     const int* rowptr,
                     int* colind, T* values,
                     T tol = 0)
{
    for(int i = 0; i < N; i++)
    {
        int idx = rowptr[i];
        for(int j = 0; j <= i; j++)     // triangle inférieur
        {
            T val = A[i*N + j];
            if (std::fabs(val) > tol)
            {
                colind[idx] = j;
                values[idx] = val;
                idx++;
            }
        }
    }
}

template<typename T>
void test() {
  const int nx = 3;
  const int N = 4;
  int size_value, size_rowptr, size_colind;


  T h_S[] = {
    1, 13, 14, 25, 28, 31, 0, 0, 0, 0, 0, 0,
    13, 2, 15, 26, 29, 32, 0, 0, 0, 0, 0, 0,
    14, 15, 3, 27, 30, 33, 0, 0, 0, 0, 0, 0,
    25, 26, 27, 4, 16, 17, 34, 37, 40, 0, 0, 0,
    28, 29, 30, 16, 5, 18, 35, 38, 41, 0, 0, 0,
    31, 32, 33, 17, 18, 6, 36, 39, 42, 0, 0, 0,
    0, 0, 0, 34, 35, 36, 7, 19, 20, 43, 46, 49,
    0, 0, 0, 37, 38, 39, 19, 8, 21, 44, 47, 50,
    0, 0, 0, 40, 41, 42, 20, 21, 9, 45, 48, 51,
    0, 0, 0, 0, 0, 0, 43, 44, 45, 10, 22, 23,
    0, 0, 0, 0, 0, 0, 46, 47, 48, 22, 11, 24,
    0, 0, 0, 0, 0, 0, 49, 50, 51, 23, 24, 12,
  };

  T gamma[] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
  };

  T lambda[] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  };


  Conjugate_Gradien<T>(h_S, gamma, lambda, nx, N);
  printVector<T>("CG", lambda, nx * N);

  float kernel_execution_time = 0.0f;
  
  T* triangular = value<T>(h_S, nx, N, size_value);
  const int* rowptr = generate_rowptr(nx, N, size_rowptr);
  const int* colind = generate_colind(nx, N, size_colind);
  printMatrix("matrix S", h_S, N*nx, 0);
  printVector("value", triangular, size_value, 0);
  printVector("row ptr", rowptr, size_rowptr, 0);
  printVector("colInd", colind, size_colind, 0);

  std::cout << "S_symetric : " << isSymmetric(h_S, N*nx) << std::endl;


  T lambda_i[] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  };

  solver_gpu<T>(triangular, gamma, lambda_i, rowptr, colind, N*nx, size_value, size_rowptr, &kernel_execution_time);

  printVector<T>("cudss", lambda_i, nx * N);

  std::cout << kernel_execution_time << std::endl;

  delete[] triangular;
  delete[] rowptr;
  delete[] colind;
}

int main() {
  test<double>();
  return 0;
}
