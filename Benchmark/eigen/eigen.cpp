#include <stdio.h>
#include <stdlib.h>
#include <cmath> 
#include <chrono>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include "utils.h"
#include "constant.h"


template <typename T>
Eigen::SparseMatrix<T> denseToSparse(const T* dense, uint32_t n)
{
    // Création en format LIL, + efficace pour remplir une sparse
    Eigen::SparseMatrix<T> A_sparse(n, n);
    std::vector<Eigen::Triplet<T>> triplets;
    triplets.reserve(n * 10);  // estimation basse, ajustée ensuite

    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < n; j++) {
            T val = dense[i * n + j];
            if (val != T(0)) {
                triplets.emplace_back(i, j, val);
            }
        }
    }

    A_sparse.setFromTriplets(triplets.begin(), triplets.end());
    return A_sparse;
}


template<typename T>
void run_benchmark(uint32_t state_size, uint32_t knot_points) {
    const uint32_t Nnx = state_size * knot_points;

    // Matrice full dense (remplie de zéros)
    T* h_S = generate_spd_block_tridiagonal<T>(state_size, knot_points);
    T* h_gamma = generate_random_vector<T>(Nnx);

    // --- Conversion dense -> sparse ---
    Eigen::SparseMatrix<T> A_sparse = denseToSparse(h_S, Nnx);

    // Vecteur RHS
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> b(h_gamma, Nnx);

    // Solveur SPD : SimplicialLDLT (le plus efficace)
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>> solver;

    auto start = std::chrono::high_resolution_clock::now();

    solver.compute(A_sparse);
    Eigen::Matrix<T, Eigen::Dynamic, 1> x = solver.solve(b);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<T, std::milli> exec_time_ms = end - start;
    std::cout << exec_time_ms.count() << std::endl;

    free(h_S);
    free(h_gamma);
}

int main() {

  const uint32_t state_size = STATE_SIZE;
  const uint32_t knot_points = KNOT_POINTS;

  #if TIME_EXECUTION_DOUBLE
    run_benchmark<double>(state_size, knot_points);
  #endif

  #if TIME_EXECUTION_FLOAT
    run_benchmark<float>(state_size, knot_points);
  #endif

  return 0;
}