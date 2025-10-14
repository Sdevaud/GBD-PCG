#pragma once
#include <iostream>
#include <cmath>



// Accès à l'élément A(i,j) pour tableau plat
inline float& el(float* A, int i, int j, int n) {
    return A[i * n + j];
}

// Gauss-Jordan pour matrice carrée n x n
void gaussJordan(float* A, float* b, float* x, int n) {
    // Création de la matrice augmentée dans un tableau temporaire
    float aug[n][n+1]; // n=6 ici, dernière colonne = b
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            aug[i][j] = el(A,i,j,n);
        }
        aug[i][n] = b[i];
    }

    // Élimination
    for (int i = 0; i < n; i++) {
        float pivot = aug[i][i];
        if (fabs(pivot) < 1e-12) {
            std::cerr << "Pivot nul → système peut-être singulier !" << std::endl;
            exit(1);
        }

        // Normalisation de la ligne pivot
        for (int j = 0; j <= n; j++)
            aug[i][j] /= pivot;

        // Élimination des autres lignes
        for (int k = 0; k < n; k++) {
            if (k == i) continue;
            float factor = aug[k][i];
            for (int j = 0; j <= n; j++)
                aug[k][j] -= factor * aug[i][j];
        }
    }

    // Récupération de la solution
    for (int i = 0; i < n; i++)
        x[i] = aug[i][n];
}

void matVecProduct(const float* A, const float* x, float* b, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            b[i] = A[i * n + j] * x[j]; // accès à l'élément (i,j)
        }
    }
}