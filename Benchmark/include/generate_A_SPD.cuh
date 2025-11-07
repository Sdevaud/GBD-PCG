#include <iostream>
#include <cmath>
#include <random>
#include <ctime>
#include <iomanip>
#include "CG_no_GPU.cuh"

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
T* extract_block(const T* A, int i, int size_bloc, int size_A) {

  T* block = new T[size_bloc * size_bloc];

  int start_row = i / size_A;
  int start_col = i % size_A;

  for (int r = 0; r < size_bloc; ++r) {
    for (int c = 0; c < size_bloc; ++c) {
      int src_index = (start_row + r) * size_A + (start_col + c);
      int dst_index = r * size_bloc + c;
      block[dst_index] = A[src_index];
    }
  }

  return block;
}

template<typename T>
T* matmul(const T* A, const T* B, int size) {
  T* C = new T[size * size];

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
T* matinv(const T* A, int size) {
    // Crée une copie locale de A car on va la modifier
    T* M = new T[size * size];
    for (int i = 0; i < size * size; ++i)
        M[i] = A[i];

    // Crée la matrice identité (pour construire l’inverse)
    T* I = new T[size * size];
    for (int i = 0; i < size * size; ++i)
        I[i] = (i / size == i % size) ? 1 : 0;

    // === Méthode de Gauss–Jordan ===
    for (int k = 0; k < size; ++k) {
        // Trouve le pivot
        T pivot = M[k * size + k];
        if (pivot == 0) {
            std::cerr << "Erreur: pivot nul à l'étape " << k << std::endl;
            delete[] M;
            delete[] I;
            return nullptr;
        }

        // Normalise la ligne du pivot
        for (int j = 0; j < size; ++j) {
            M[k * size + j] /= pivot;
            I[k * size + j] /= pivot;
        }

        // Élimine les autres lignes
        for (int i = 0; i < size; ++i) {
            if (i == k) continue;
            T factor = M[i * size + k];
            for (int j = 0; j < size; ++j) {
                M[i * size + j] -= factor * M[k * size + j];
                I[i * size + j] -= factor * I[k * size + j];
            }
        }
    }

    delete[] M;
    return I;
}

template<typename T>
T* mat_transpose(const T* A, int rows, int cols) {
    T* AT = new T[rows * cols];

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            AT[j * rows + i] = A[i * cols + j];
        }
    }

    return AT;
}


template<typename T>
T* compute_D1inv_O1_D2inv(const T* D1, const T* D2, const T* O1, int nx) {
  T* D1_inv = matinv<T>(D1, nx);
  T* D2_inv = matinv<T>(D2, nx);
  T* OD = matmul<T>(O1, D2_inv, nx);
  T* DOD = matmul<T>(D1_inv, OD, nx);

  for (int i = 0; i < nx*nx; ++i) DOD[i] *=  -1;

  delete[] D1_inv;
  delete[] D2_inv;
  delete[] OD;

  return DOD;
}

template<typename T>
void copy_block_in_matrix(T* Pinv, T* block, int size_bloc, int index) {
for (int i = 0; i < size_bloc; ++i) {
    Pinv[index + i] = block[i];
  }
}

template<typename T>
T* formPolyPreconditioner_Pinv(T* S, int N, int nx) {

  int size_P = 3 * nx * nx * N;
  T* Pinv = (T*)calloc(size_P, sizeof(T));

  for (int i = 0; i < N-1; ++i) {
    T* Di = extract_block<T>(S, i*nx*nx*N + i*nx, nx, N * nx);
    T* Oi = extract_block<T>(S, i*nx*nx*N + i*nx + nx, nx, N * nx);
    T* Di1 = extract_block<T>(S, (i+1)*nx*nx*N + (i+1)*nx, nx, N * nx);
    T* Di_inv = matinv<T>(Di, nx);
    T* DOD = compute_D1inv_O1_D2inv<T>(Di, Di1, Oi, nx);
    T* DOD_T = mat_transpose<T>(DOD, nx, nx);

    copy_block_in_matrix<T>(Pinv, Di_inv, nx*nx, 3*i*nx*nx + nx*nx);
    copy_block_in_matrix<T>(Pinv, DOD_T, nx*nx, 3*i*nx*nx + 2*nx*nx);
    copy_block_in_matrix<T>(Pinv, DOD, nx*nx, 3*i*nx*nx + 3*nx*nx);

    delete[] Di;
    delete[] Oi;
    delete[] Di1;
    delete[] Di_inv;
    delete[] DOD;
    delete[] DOD_T;
  }
  T* DN = extract_block<T>(S, (N-1)*nx*nx*N + (N-1)*nx, nx, N * nx);
  T* DN_inv = matinv<T>(DN, nx);
  copy_block_in_matrix<T>(Pinv, DN_inv, nx*nx, 3*(N-1)*nx*nx + nx*nx);

  delete[] DN;
  delete[] DN_inv;

  return Pinv;
}

template<typename T>
T* formPolyPreconditioner_H(T* S, int N, int nx) {

  // Chaque bloc fait nx*nx, chaque étage (bloc tri) fait 3*nx*nx,
  // mais H est penta-diagonale, donc aussi 3*N blocs principaux (comme Pss)
  int size_H = 3 * nx * nx * N;
  T* H = (T*)calloc(size_H, sizeof(T));

  // On aura besoin du "O_add" (équivalent du formPreconditionerSS)
  // donc on commence par calculer O_add à partir de S
  // (exactement comme dans ton formPolyPreconditioner_Pinv)
  for (int i = 0; i < N - 1; ++i) {

    // Extraction des blocs Di, Di+1 et Oi
    T* Di = extract_block<T>(S, i * nx * nx * N + i * nx, nx, N * nx);
    T* Oi = extract_block<T>(S, i * nx * nx * N + i * nx + nx, nx, N * nx);
    T* Di1 = extract_block<T>(S, (i + 1) * nx * nx * N + (i + 1) * nx, nx, N * nx);

    // Inversion des blocs diagonaux
    T* Di_inv = matinv<T>(Di, nx);
    T* Di1_inv = matinv<T>(Di1, nx);

    // Calcul de O_add = -Di^{-1} * Oi * Di1^{-1}
    T* tmp = matmul<T>(Di_inv, Oi, nx);
    T* O_add = matmul<T>(tmp, Di1_inv, nx);
    for (int k = 0; k < nx * nx; ++k) O_add[k] = -O_add[k];

    delete[] tmp;
    delete[] Di_inv;
    delete[] Di1_inv;

    // Maintenant on peut calculer les blocs de H selon le formalisme MATLAB
    if (i == 0) {
      // Premier bloc diagonal : D_H{1} = -O_add{1} * O{1}'
      T* Oi_T = mat_transpose<T>(Oi, nx, nx);
      T* DiagH = matmul<T>(O_add, Oi_T, nx);
      for (int k = 0; k < nx * nx; ++k) DiagH[k] = -DiagH[k];
      copy_block_in_matrix<T>(H, DiagH, nx * nx, 3 * i * nx * nx + nx * nx);
      delete[] DiagH;
      delete[] Oi_T;
    }

    if (i == N - 2) {
      // Dernier bloc diagonal : D_H{N} = -O_add{N-1}' * O{N-1}
      T* O_add_T = mat_transpose<T>(O_add, nx, nx);
      T* DiagH = matmul<T>(O_add_T, Oi, nx);
      for (int k = 0; k < nx * nx; ++k) DiagH[k] = -DiagH[k];
      copy_block_in_matrix<T>(H, DiagH, nx * nx, 3 * (i + 1) * nx * nx + nx * nx);
      delete[] O_add_T;
      delete[] DiagH;
    }

    // Pour les blocs internes (2 ≤ i ≤ N−1)
    if (i >= 1) {
      // On a besoin de O_add(i-1) et O(i-1)
      T* O_prev = extract_block<T>(S, (i - 1) * nx * nx * N + (i - 1) * nx + nx, nx, N * nx);
      T* Di_prev = extract_block<T>(S, (i - 1) * nx * nx * N + (i - 1) * nx, nx, N * nx);
      T* Di_prev_inv = matinv<T>(Di_prev, nx);
      T* tmp_prev = matmul<T>(Di_prev_inv, O_prev, nx);
      T* O_add_prev = matmul<T>(tmp_prev, Di_inv, nx);
      for (int k = 0; k < nx * nx; ++k) O_add_prev[k] = -O_add_prev[k];

      // D_H{i} = -O_add{i} * O{i}' - O_add{i-1}' * O{i-1}
      T* Oi_T = mat_transpose<T>(Oi, nx, nx);
      T* Oprev_T = mat_transpose<T>(O_prev, nx, nx);
      T* term1 = matmul<T>(O_add, Oi_T, nx);
      T* term2_tmp = mat_transpose<T>(O_add_prev, nx, nx);
      T* term2 = matmul<T>(term2_tmp, Oprev_T, nx);
      for (int k = 0; k < nx * nx; ++k) term1[k] = -(term1[k] + term2[k]);
      copy_block_in_matrix<T>(H, term1, nx * nx, 3 * i * nx * nx + nx * nx);

      // O_up2{i-1} = -O_add{i-1} * O{i}
      T* O_up2 = matmul<T>(O_add_prev, Oi, nx);
      for (int k = 0; k < nx * nx; ++k) O_up2[k] = -O_up2[k];
      copy_block_in_matrix<T>(H, O_up2, nx * nx, 3 * (i - 1) * nx * nx + 3 * nx * nx);

      // O_down2{i-1} = -O_add{i}' * O{i-1}'
      T* O_down2_tmp = mat_transpose<T>(O_add, nx, nx);
      T* O_down2 = matmul<T>(O_down2_tmp, Oprev_T, nx);
      for (int k = 0; k < nx * nx; ++k) O_down2[k] = -O_down2[k];
      copy_block_in_matrix<T>(H, O_down2, nx * nx, 3 * (i - 1) * nx * nx + nx * nx);

      delete[] O_prev;
      delete[] Di_prev;
      delete[] Di_prev_inv;
      delete[] tmp_prev;
      delete[] O_add_prev;
      delete[] Oi_T;
      delete[] Oprev_T;
      delete[] term1;
      delete[] term2_tmp;
      delete[] term2;
      delete[] O_up2;
      delete[] O_down2;
    }

    delete[] Di;
    delete[] Oi;
    delete[] Di1;
    delete[] O_add;
  }

  return H;
}
