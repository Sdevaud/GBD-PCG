#include <iostream>
#include <cmath>
#include <random>

// Génère une matrice bloc-tridiagonale SPD (float* A)
float* generate_spd_block_tridiagonal(int state, int horizon, unsigned int seed = 42) {
    int N = state;       // taille des blocs
    int n = horizon;     // nombre de blocs
    int dim = N * n;     // taille totale de la matrice (dim x dim)

    // Allocation mémoire (1D)
    float* A = new float[dim * dim];
    for (int i = 0; i < dim * dim; ++i)
        A[i] = 0.0f;

    // Générateur aléatoire
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Construction bloc par bloc
    for (int k = 0; k < n; ++k) {
        // ---- Bloc diagonal D_k ----
        std::vector<float> R(N * N);
        for (int i = 0; i < N * N; ++i)
            R[i] = dist(gen);

        // D_k = R^T * R + N * I
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int t = 0; t < N; ++t)
                    sum += R[t * N + i] * R[t * N + j]; // R^T * R
                if (i == j)
                    sum += N;  // SPD renforcement
                A[(k*N + i) * dim + (k*N + j)] = sum;
            }
        }

        // ---- Bloc hors-diagonal O_k ----
        if (k < n - 1) {
            std::vector<float> O(N * N);
            for (int i = 0; i < N * N; ++i)
                O[i] = 0.1f * dist(gen); // petite magnitude

            // Remplir O_k et O_k^T dans A
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    A[(k*N + i) * dim + ((k+1)*N + j)] = O[i*N + j];  // O_k
                    A[((k+1)*N + j) * dim + (k*N + i)] = O[j*N + i];  // O_k^T
                }
            }
        }
    }

    // ---- SPD global (petite régularisation) ----
    for (int i = 0; i < dim; ++i)
        A[i * dim + i] += 1e-3f;

    return A; // pointeur sur la matrice 1D (à libérer avec delete[])
}

// Génère un vecteur b aléatoire de taille 'dim' et retourne un pointeur vers un tableau float[]
float* generate_random_vector(int dim, unsigned int seed = 42) {
    // Allocation dynamique
    float* b = new float[dim];

    // Générateur pseudo-aléatoire
    std::mt19937 gen(seed);                     // moteur Mersenne Twister
    std::normal_distribution<float> dist(0.0f, 1.0f);  // distribution normale (moyenne=0, écart-type=1)

    // Remplissage du vecteur
    for (int i = 0; i < dim; ++i) {
        b[i] = dist(gen);
    }

    return b; // pointeur vers le tableau
}