#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

using Matrix = std::vector<std::vector<double>>;

// Fonction Gauss-Jordan pour résoudre Ax = b
// A est une matrice carrée n x n
// b est un vecteur n x 1
// Retourne le vecteur solution x
std::vector<double> gauss_jordan(Matrix A, std::vector<double> b) {
    int n = A.size();
    assert(b.size() == n);

    for (int i = 0; i < n; ++i) {
        // Pivot partiel : trouver la ligne avec le plus grand pivot
        int maxRow = i;
        for (int k = i+1; k < n; ++k)
            if (std::fabs(A[k][i]) > std::fabs(A[maxRow][i]))
                maxRow = k;

        std::swap(A[i], A[maxRow]);
        std::swap(b[i], b[maxRow]);

        // Normaliser la ligne pivot
        double pivot = A[i][i];
        assert(pivot != 0); // matrice non singulière
        for (int j = i; j < n; ++j)
            A[i][j] /= pivot;
        b[i] /= pivot;

        // Élimination des autres lignes
        for (int k = 0; k < n; ++k) {
            if (k == i) continue;
            double factor = A[k][i];
            for (int j = i; j < n; ++j)
                A[k][j] -= factor * A[i][j];
            b[k] -= factor * b[i];
        }
    }
  return b; // le vecteur solution x
}