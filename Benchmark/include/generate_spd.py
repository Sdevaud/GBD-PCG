import numpy as np
from pathlib import Path
import glob


# ---------------------------
# utilitaires
# ---------------------------

def unitary_matrix(n: int) -> np.ndarray:
    """
    Approximation de unitaryMatrix(nx) :
    on prend une matrice gaussienne et on fait une décomposition QR.
    Q est orthonormale (O(n)).
    """
    M = np.random.randn(n, n)
    Q, _ = np.linalg.qr(M)
    return Q


def random_spd(n: int) -> np.ndarray:
    """
    Renvoie une matrice symétrique définie positive :
    Q = T * diag(rand) * T'
    """
    T = unitary_matrix(n)
    d = np.random.rand(n)  # valeurs propres > 0
    return T @ np.diag(d) @ T.T


def random_diag(n: int) -> np.ndarray:
    """
    Renvoie diag(rand(n,1)) : matrice diagonale positive.
    """
    return np.diag(np.random.rand(n))


# ---------------------------
# stub form_kkt_schur (équivalent de formKKTSchur)
# ---------------------------

def form_kkt_schur(A, B, Q, R, N):
    """
    A, B, Q, R : listes de np.ndarray (équivalent des cell arrays MATLAB)
    [D, O, S] = formKKTSchur(A, B, Q, R, N)

    À remplir avec ton vrai calcul. Ici on renvoie juste des None
    / structures vides pour illustrer.
    """
    D = []   # à remplacer
    O = []   # à remplacer
    S = None # à remplacer
    return D, O, S


def main():
    # Paramètres
    N  = 10   # horizon
    nx = 4    # dim. état
    nu = 2    # dim. commande

    A = [np.random.rand(nx, nx) for _ in range(N - 1)]
    B = [np.random.rand(nx, nu) for _ in range(N - 1)]

    Q = [None] * N
    R = [None] * (N - 1)

    for k in range(N - 1):
        Q[k] = random_spd(nx)
        R[k] = random_diag(nu)

    # Dernier Q_N
    Q[N - 1] = random_spd(nx)

    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir()
    else:
        for f in glob.glob(str(data_dir / "I_H*")):
            Path(f).unlink()

    save_path = data_dir / "problem_data.npz"
    np.savez(
        save_path,
        A=np.array(A, dtype=object),
        B=np.array(B, dtype=object),
        Q=np.array(Q, dtype=object),
        R=np.array(R, dtype=object),
        N=N,
        nx=nx,
        nu=nu,
    )
    print(f"Dataset sauvegardé dans {save_path}")

    # --------
    # Lecture du dataset (je crée un data set et je le lis, ça marche aussi)
    # --------
    loaded = np.load(save_path, allow_pickle=True)

    A_loaded = list(loaded["A"])
    B_loaded = list(loaded["B"])
    Q_loaded = list(loaded["Q"])
    R_loaded = list(loaded["R"])
    N_loaded = int(loaded["N"])
    nx_loaded = int(loaded["nx"])
    nu_loaded = int(loaded["nu"])

    print(f"Dataset relu : N={N_loaded}, nx={nx_loaded}, nu={nu_loaded}")
    print("A[0] shape =", A_loaded[0].shape)
    print("Q[N-1] shape =", Q_loaded[-1].shape)

    # --------
    # Calcul de D, O, S via form_kkt_schur (stub)
    # --------
    D, O, S = form_kkt_schur(A_loaded, B_loaded, Q_loaded, R_loaded, N_loaded)
    print("form_kkt_schur appelé (stub). D, O, S =", D, O, S)


if __name__ == "__main__":
    main()
