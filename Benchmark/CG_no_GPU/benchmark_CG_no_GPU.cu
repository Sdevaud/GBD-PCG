#include <stdio.h>
#include <stdlib.h>
#include <cmath> 
#include <chrono>
#include "CG_no_GPU.cuh"
#include "generate_A_SPD.cuh"

int main(int argc, char* argv[]) {
    // Vérifie les arguments
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <state> <horizon>" << std::endl;
        return 1;
    }

    // Récupération des arguments
    int state   = std::atoi(argv[1]);
    int horizon = std::atoi(argv[2]);
    int N = state * horizon;  // taille du système total

    // Génération des matrices
    float* A = generate_spd_block_tridiagonal(state, horizon);
    float* B = generate_random_vector(state * horizon);
    float* C = (float*)malloc(N * sizeof(float));

    // Initialisation de C
    for (int i = 0; i < N; ++i)
        C[i] = 0.0f;

    // --- Début du chronométrage ---
    auto start = std::chrono::high_resolution_clock::now();

    Conjugate_Gradien(A, B, C, state, horizon);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> exec_time = end - start;

    // --- Fin du chronométrage ---

    // Affiche uniquement le temps d'exécution en millisecondes (pour Python)
    std::cout << exec_time.count() << std::endl;

    // Libération mémoire
    free(A);
    free(B);
    free(C);

    return 0;
}