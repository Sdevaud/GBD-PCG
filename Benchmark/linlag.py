import numpy as np
import sys
import time
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

def generate_spd_block_tridiagonal_sparse(state, horizon, seed=None):
    """
    Génère une matrice bloc-tridiagonale symétrique définie positive (SPD)
    de taille (state * horizon) × (state * horizon).

    Structure :
        [ D1   O1^T   0      ...       0     ]
        [ O1    D2    O2^T   ...       0     ]
        [ 0     O2    D3     ...       0     ]
        [ ...   ...   ...     ...     On-1^T ]
        [ 0     0     0      On-1     Dn    ]

    Où chaque bloc Dk, Ok ∈ R^(state × state).
    """
    if seed is not None:
      np.random.seed(seed)

    N = state
    n = horizon
    dim = N * n

    A = lil_matrix((dim, dim))  # Sparse modifiable

    for k in range(n):
        # Bloc diagonal D_k (SPD)
        R = np.random.randn(N, N)
        Dk = R.T @ R + N * np.eye(N)

        A[k*N:(k+1)*N, k*N:(k+1)*N] = Dk

        # Bloc hors-diagonal O_k
        if k < n - 1:
            Ok = np.random.randn(N, N) * 0.1
            A[k*N:(k+1)*N, (k+1)*N:(k+2)*N] = Ok
            A[(k+1)*N:(k+2)*N, k*N:(k+1)*N] = Ok.T

    # Renforce SPD
    A += 1e-3 * np.eye(dim)

    return A.tocsc()  # format efficace pour spsolve()

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 linlasolve.py <state> <horizon>")
        sys.exit(1)

    state = int(sys.argv[1])
    horizon = int(sys.argv[2])

    # Génération sparse de A
    A = generate_spd_block_tridiagonal_sparse(state, horizon)
    b = np.random.randn(state * horizon)

    # Résolution sparse + timing
    start = time.perf_counter()
    x = spsolve(A, b)
    end = time.perf_counter()

    exec_time_ms = (end - start) * 1000
    print(f"{exec_time_ms:.6f}")


def show_matrix(A, precision=2):
    """
    Affiche toute la matrice A sans troncature :
      - toutes les lignes et colonnes sont visibles
      - les nombres sont arrondis pour plus de lisibilité
    """
    np.set_printoptions(
        precision=precision,     # nombre de décimales
        suppress=True,           # pas de notation scientifique inutile
        linewidth=200,           # largeur max d'affichage par ligne
        threshold=np.inf         # <-- affiche toute la matrice !
    )

    print(f"Matrice A (taille: {A.shape[0]}x{A.shape[1]}):\n")
    print(A)

    # Réinitialiser les options après affichage
    np.set_printoptions()

if __name__ == "__main__":
    main()

