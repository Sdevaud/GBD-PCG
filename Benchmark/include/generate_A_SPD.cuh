#include <iostream>
#include <cmath>
#include <random>
#include <ctime>
#include <iomanip>

using namespace std;

template<typename T>
T* transform_matrix(const T* Matrix, const int state_size, const int knot_points)
{
    const int N = knot_points;            // number of block rows/cols
    const int nx = state_size;
    const int nx2 = nx * nx;
    const int dim = N * nx;               // total dimension = N*nx

    const int total_blocks = 3 * N;       // 0, D1, O1ᵀ, O1, ..., DN, 0
    T* h_S = new T[total_blocks * nx2];

    // Copy an nx×nx block from Matrix(row_block, col_block) into block index b
    auto copy_block = [&](int block_idx, int row_block, int col_block) {
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < nx; ++j) {
                int src = (row_block * nx + i) * dim + (col_block * nx + j);
                int dst = block_idx * nx2 + (i * nx + j);
                h_S[dst] = Matrix[src];
            }
        }
    };

    // ---- 1) first block = 0 ----
    for (int k = 0; k < nx2; k++) {
        h_S[k] = 0.0;
    }

    int b = 1; // block counter

    // ---- 2) Fill blocks ----
    for (int k = 0; k < N; ++k) {
        // D_k
        copy_block(b++, k, k);

        if (k < N-1) {
            // O_k^T
            copy_block(b++, k+1, k);
            // O_k
            copy_block(b++, k, k+1);
        }
    }

    // ---- 3) last block = 0 ----
    for (int k = 0; k < nx2; k++) {
        h_S[(total_blocks-1) * nx2 + k] = 0.0;
    }

    return h_S;
}



template<typename T>
T norm_vecotr(const T* vector, const int size) {
  T norm = 0;
  for(int i = 0; i < size; ++i) {
    norm += vector[i] * vector[i];
  }

  return sqrt(norm);
}

template<typename T>
void printVector(string vector_name, const T* arr, int size, int precision = 4) {
  std::cout << vector_name << " : ";  
  std::cout << "[";
    for (int i = 0; i < size; i++) {
        cout << fixed << setprecision(precision) << arr[i];
        if (i != size - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

template<typename T>
void printMatrix(string matrix_name, const T* mat, int size, int precision = 4) {
  std::cout << matrix_name << " : \n";
  for (int i = 0; i < size; i++) {
      std::cout << "[";
      for (int j = 0; j < size; j++) {
          std::cout << fixed << setprecision(precision) << mat[i * size + j];
          if (j != size - 1) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
  }
}

template<typename T>
void mat_mul_vector(const T* matrix, const T* vector, T* result, int size) {
  for (int i = 0; i < size; ++i) {
    for(int j = 0; j < size; ++j) {
      result[i] += matrix[i*size + j] * vector[j];
    }
  }
}

// Generates a block-tridiagonal SPD (T* A) matrix
template<typename T>
T* generate_spd_block_tridiagonal(int state_size, int knot_points, unsigned int seed = 0) {

    if (seed == 0)
      seed = static_cast<unsigned int>(std::time(nullptr));
    std::mt19937 gen(seed);
    std::normal_distribution<T> dist(0.0f, 1.0f);


    int N = state_size;
    int n = knot_points;
    int dim = N * n;


    T* A = new T[dim * dim];
    for (int i = 0; i < dim * dim; ++i)
        A[i] = 0.0f;

    // Building bloc per Bloc
    for (int k = 0; k < n; ++k) {
        // ---- Bloc diagonal D_k ----
        std::vector<T> R(N * N);
        for (int i = 0; i < N * N; ++i)
            R[i] = dist(gen);

        // D_k = R^T * R + N * I
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                T sum = 0.0f;
                for (int t = 0; t < N; ++t)
                    sum += R[t * N + i] * R[t * N + j]; // R^T * R
                if (i == j)
                    sum += N;  // SPD reinforcement
                A[(k*N + i) * dim + (k*N + j)] = sum;
            }
        }

        // ---- Bloc O_k ----
        if (k < n - 1) {
            std::vector<T> O(N * N);
            for (int i = 0; i < N * N; ++i)
                O[i] = 0.1f * dist(gen);

            // fill A
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    A[(k*N + i) * dim + ((k+1)*N + j)] = O[i*N + j];  // O_k
                    A[((k+1)*N + j) * dim + (k*N + i)] = O[i*N + j];  // O_k^T
                }
            }
        }
    }

    // ---- SPD global ----
    for (int i = 0; i < dim; ++i)
        A[i * dim + i] += 1e-3f;

    return A;
}

// Generates a vector (T* b) 
template<typename T>
T* generate_random_vector(int dim, unsigned int seed = 0) {
  
    if (seed == 0)
      seed = static_cast<unsigned int>(std::time(nullptr));
    std::mt19937 gen(seed);
    std::normal_distribution<T> dist(0.0f, 1.0f);

    T* b = new T[dim];

    for (int i = 0; i < dim; ++i) {
        b[i] = dist(gen);
    }

    return b;
}

template<typename T>
T* block_ptr(T* A, int i, int j, int N, int nx) {
    return A + (i*N + j) * nx * nx;
}

template<typename T>
void mat_copy(T* dst, const T* src, int nx) {
    memcpy(dst, src, nx*nx*sizeof(T));
}

template<typename T>
void mat_zero(T* A, int nx) {
    memset(A, 0, nx*nx*sizeof(T));
}

template<typename T>
void mat_transpose(T* AT, const T* A, int nx) {
    for(int i=0;i<nx;i++)
        for(int j=0;j<nx;j++)
            AT[j*nx+i] = A[i*nx+j];
}

template<typename T>
void mat_mul(T* C, const T* A, const T* B, int nx) {
    for(int i=0;i<nx;i++)
        for(int j=0;j<nx;j++){
            T s = 0;
            for(int k=0;k<nx;k++) s += A[i*nx+k] * B[k*nx+j];
            C[i*nx+j] = s;
        }
}

template<typename T>
void mat_inv(T* Ainv, const T* A, int nx) {
    gauss_jordan(Ainv, A, nx); 
}

template<typename T>
void mat_add(T* C, const T* A, const T* B, int nx) {
    for(int i=0;i<nx*nx;i++) C[i] = A[i] + B[i];
}

template<typename T>
void mat_sub(T* C, const T* A, const T* B, int nx) {
    for(int i=0;i<nx*nx;i++) C[i] = A[i] - B[i];
}

template<typename T>
void formPreconditioner_h_Pinv(T* S, T* P, int N, int nx) {
  // P same layout N×N blocks, input S block-tridiag, output P block-tridiag
  for(int i=0;i<N;i++)
      for(int j=0;j<N;j++)
          mat_zero(block_ptr(P,i,j,N,nx), nx);

  for(int i=0;i<N;i++) {
      T* Di  = block_ptr(S, i, i, N, nx);
      T* Di_inv = (T*)malloc(nx*nx*sizeof(T));
      mat_inv(Di_inv, Di, nx);

      // place inv(Di) on diagonal block
      mat_copy(block_ptr(P,i,i,N,nx), Di_inv, nx);

      if(i < N-1) {
          T* Oi = block_ptr(S, i, i+1, N, nx);
          T *tmp = (T*)malloc(nx*nx*sizeof(T));
          T *tmp2 = (T*)malloc(nx*nx*sizeof(T));
          T* Dnext = block_ptr(S, i+1,i+1,N,nx);
          T* Dnext_inv = (T*)malloc(nx*nx*sizeof(T));
          mat_inv(Dnext_inv, Dnext, nx);

          // O_P{i} = -inv(D{i})*O{i}*inv(D{i+1})
          mat_mul(tmp, Di_inv, Oi, nx);
          mat_mul(tmp2, tmp, Dnext_inv, nx);

          T* OP = block_ptr(P, i, i+1, N, nx);
          for(int k=0;k<nx*nx;k++) OP[k] = -tmp2[k];

          free(tmp); free(tmp2); free(Dnext_inv);
      }
      free(Di_inv);
  }
}

template<typename T>
void formPolyPreconditionerH(T* S, T* H, int N, int nx) {
  // Step 1: build SS preconditioner P, but we only need O_add (off-diag)
  T* P = (T*)calloc(N*N*nx*nx, sizeof(T));
  formPreconditionerSS(S, P, N, nx);

  // H initially zero full pentadiagonal
  memset(H, 0, N*N*nx*nx*sizeof(T));

  T *tmp = (T*)malloc(nx*nx*sizeof(T));
  T *tmp2 = (T*)malloc(nx*nx*sizeof(T));
  T *OiT = (T*)malloc(nx*nx*sizeof(T));
  T *OaddT = (T*)malloc(nx*nx*sizeof(T));

  for(int i=0;i<N;i++) {
      T* DHi = block_ptr(H, i, i, N, nx);
      if(i == 0) {
          T* Oadd = block_ptr(P,0,1,N,nx);
          T* O = block_ptr(S,0,1,N,nx);
          mat_transpose(OiT, O, nx);
          mat_mul(tmp, Oadd, OiT, nx);
          for(int k=0;k<nx*nx;k++) DHi[k] = -tmp[k];
      }
      else if(i == N-1) {
          T* Oadd = block_ptr(P, N-2, N-1, N, nx);
          T* O = block_ptr(S, N-2, N-1, N, nx);
          mat_transpose(OaddT, Oadd, nx);
          mat_mul(tmp, OaddT, O, nx);
          for(int k=0;k<nx*nx;k++) DHi[k] = -tmp[k];
      }
      else {
          // D_H{i}
          T* Oadd_i = block_ptr(P,i,i+1,N,nx);
          T* O_i    = block_ptr(S,i,i+1,N,nx);
          T* Oadd_im1 = block_ptr(P,i-1,i,N,nx);
          T* O_im1    = block_ptr(S,i-1,i,N,nx);

          mat_transpose(OiT, O_i, nx);
          mat_mul(tmp, Oadd_i, OiT, nx);       // A
          mat_transpose(OaddT, Oadd_im1, nx);
          mat_mul(tmp2, OaddT, O_im1, nx);     // B

          T* DHi = block_ptr(H, i,i,N,nx);
          for(int k=0;k<nx*nx;k++) 
              DHi[k] = -(tmp[k] + tmp2[k]);

          // O_up2(i-1) = -O_add(i-1)*O(i)
          T* Up2 = block_ptr(H, i-1, i+1, N, nx);
          mat_mul(tmp, Oadd_im1, O_i, nx);
          for(int k=0;k<nx*nx;k++) Up2[k] = -tmp[k];

          // O_down2 = -O_add(i)'*O(i-1)'
          T* Down2 = block_ptr(H, i+1, i-1, N, nx);
          mat_transpose(OiT, O_i, nx);
          mat_mul(tmp, OaddT, O_im1, nx);
          for(int k=0;k<nx*nx;k++) Down2[k] = -tmp[k];
      }
  }

  free(tmp); free(tmp2); free(OiT); free(OaddT);
}
