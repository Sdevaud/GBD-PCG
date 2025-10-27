#include <iostream>
#include <cmath>
#include <cassert>
#include <algorithm>

// -----------------------------------------------------------------------------
// Fonction Gauss–Jordan : résout Ax = b
// Entrées :
//   - A : tableau 1D de taille n*n (row-major)
//   - b : tableau 1D de taille n
//   - n : dimension du système
// Sortie :
//   - x (écrit dans b à la fin)
// -----------------------------------------------------------------------------
void gauss_jordan(double* A, double* b, int n)
{
    for (int i = 0; i < n; ++i) {
        int maxRow = i;
        double maxVal = std::fabs(A[i * n + i]);
        for (int k = i + 1; k < n; ++k) {
            double val = std::fabs(A[k * n + i]);
            if (val > maxVal) {
                maxVal = val;
                maxRow = k;
            }
        }

        if (maxRow != i) {
            for (int j = 0; j < n; ++j)
                std::swap(A[i * n + j], A[maxRow * n + j]);
            std::swap(b[i], b[maxRow]);
        }

        double pivot = A[i * n + i];
        assert(pivot != 0.0 && "Matrice singulière !");
        for (int j = i; j < n; ++j)
            A[i * n + j] /= pivot;
        b[i] /= pivot;

        for (int k = 0; k < n; ++k) {
            if (k == i) continue;
            double factor = A[k * n + i];
            for (int j = i; j < n; ++j)
                A[k * n + j] -= factor * A[i * n + j];
            b[k] -= factor * b[i];
        }
    }
}
