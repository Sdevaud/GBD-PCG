#pragma once
#include <iostream>
#include <cmath>
#include <random>
#include <ctime>
#include <iomanip>
#include <limits>
using namespace std;

// ---------------- Function for READ from a .txt file ----------------

void readArrayFromFile(uint32_t size, const char *filename,
                       double *matrix) {
  FILE *myFile;
  myFile = fopen(filename, "r");
  if (myFile == NULL) {
    printf("Error Reading File\n");
    exit(0);
  }

  for (uint32_t i = 0; i < size; i++) {
    int ret = fscanf(myFile, "%lf,", &matrix[i]); // for double
    if (ret != 1) {
      fprintf(stderr, "Error reading at index %u\n", i);
      exit(EXIT_FAILURE);
    }
  }

  fclose(myFile);
  return;
}

void readArrayFromFile(uint32_t size, const char *filename,
                       float *matrix) {
  FILE *myFile;
  myFile = fopen(filename, "r");
  if (myFile == NULL) {
    printf("Error Reading File\n");
    exit(0);
  }

  for (uint32_t i = 0; i < size; i++) {
    int ret = fscanf(myFile, "%f,", &matrix[i]); // for float
    if (ret != 1) {
      fprintf(stderr, "Error reading at index %u\n", i);
      exit(EXIT_FAILURE);
    }
  }

  fclose(myFile);
  return;
}

// ---------------- Mathematic Tool function ----------------
template<typename T>
T vector_norm(const T* vector, const int size) {
  T norm = 0;
  for(int i = 0; i < size; ++i) {
    norm += vector[i] * vector[i];
  }

  return sqrt(norm);
}

template<typename T>
void mat_mul_vector(const T* A, const T* b, T* Ab, uint32_t size) {
  for (uint32_t i = 0; i < size; ++i) {
    for(uint32_t j = 0; j < size; ++j) {
      Ab[i] += A[i*size + j] * b[j];
    }
  }
}

template<typename T>
T* matmul(const T* A, const T* B, int size) {
  T* C = (T*) calloc(size*size, sizeof(T));

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      T sum = 0;
      for (int k = 0; k < size; ++k) {
        sum += A[i * size + k] * B[k * size + j];
      }
      C[i * size + j] = sum;
    }
  }

  return C;
}

template<typename T>
T* mat_transpose(const T* A, uint32_t rows, uint32_t cols) {
  T* AT = (T*) calloc(rows * cols, sizeof(T));

  for (uint32_t i = 0; i < rows; ++i) {
    for (uint32_t j = 0; j < cols; ++j) {
      AT[j * rows + i] = A[i * cols + j];
    }
  }

  return AT;
}

template<typename T>
void error_L2(const T* A, const T* b, 
      const T* x, const uint32_t size, T& error) {

  T* Ax= (T*)calloc(size, sizeof(T));
  mat_mul_vector(A, x, Ax, size);
  for (uint32_t i = 0; i < size; ++i) {
    error += pow(Ax[i] - b[i], 2);
  }

  error = sqrt(error);
  free(Ax);
}

template<typename T>
void is_spd(const T* A, int N) {
    bool test = true;
    T* L= (T*)calloc(N*N, sizeof(T));
    for (int i = 0; i < N * N; ++i)
        L[i] = A[i];

    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            if (std::abs(L[i*N + j] - L[j*N + i]) > 1e-6) {
                delete[] L;
                test = false;
            }
        }
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j <= i; ++j) {
            T sum = L[i*N + j];

            for (int k = 0; k < j; ++k)
                sum -= L[i*N + k] * L[j*N + k];

            if (i == j) {
                if (sum <= (T)0) {
                    delete[] L;
                    test = false;
                }
                L[i*N + j] = std::sqrt(sum);
            } else {
                L[i*N + j] = sum / L[j*N + j];
            }
        }
    }

    delete[] L;
    if (test) std::cout << "SPD \n";
    else std::cout << "no SPD \n";
}

// ---------------- Tool function for print ----------------

template<typename T>
void print_error(const T& error) {
    std::cout << std::setprecision(std::numeric_limits<T>::max_digits10)
              << error << std::endl;
}

template<typename T>
void printVector(string vector_name, const T* b, uint32_t size, int precision = 4) {
  std::cout << vector_name << " : ";  
  std::cout << "[";
    for (uint32_t i = 0; i < size; i++) {
        cout << fixed << setprecision(precision) << b[i];
        if (i != size - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

template<typename T>
void printMatrix(string matrix_name, const T* A, uint32_t size, int precision = 4) {
  std::cout << matrix_name << " : \n";
  for (uint32_t i = 0; i < size; i++) {
      std::cout << "[";
      for (uint32_t j = 0; j < size; j++) {
          std::cout << fixed << setprecision(precision) << A[i * size + j];
          if (j != size - 1) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
  }
}

// ---------------- Tool function for CUDA solver ----------------

int* generate_rowptr(const int nx, const int N, int& size)
{
  const int Nnx = N * nx;
  size = Nnx + 1;
  int* rowptr = (int*)calloc(size, sizeof(int));
  rowptr[0] = 0;

  for (int b = 0; b < N; ++b) {
    for (int j = 0; j < nx; ++j) {
      const int row = b * nx + j;
      const int nnz_row = (b == 0) ? (j + 1) : (nx + (j + 1));
      rowptr[row + 1] = rowptr[row] + nnz_row;
    }
  }
  return rowptr;
}


int* generate_colind(const int nx, const int N, int& size, const int* rowptr, const int rowptr_size)
{
  const int Nnx = N * nx;
  size = rowptr[Nnx];

  int* colind = (int*)calloc(size, sizeof(int));

  for (int b = 0; b < N; ++b) {
    for (int j = 0; j < nx; ++j) {
      const int row = b * nx + j;
      int p = rowptr[row];

      if (b > 0) {
        const int base_prev = (b - 1) * nx;
        for (int c = 0; c < nx; ++c) colind[p++] = base_prev + c;
      }

      const int base_cur = b * nx;
      for (int c = 0; c <= j; ++c) colind[p++] = base_cur + c;
    }
  }

  return colind;
}

template<typename T>
T* value(const T* h_S, const int nx, const int N, int& size, const int* rowptr, const int* colind, const int rowptr_size, const int colind_size)
{
  const int Nnx = N * nx;
  size = rowptr[Nnx];

  T* val = (T*)calloc(size, sizeof(T));

  for (int row = 0; row < Nnx; ++row) {
    for (int p = rowptr[row]; p < rowptr[row + 1]; ++p) {
      const int col = colind[p];
      val[p] = h_S[row * Nnx + col];
    }
  }

  return val;
}
