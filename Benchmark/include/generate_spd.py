import Benchmark.numpy_method as np
import os
from numpy.linalg import inv, qr
import sys

np.set_printoptions(precision=16, suppress=True)

# generate random positive definite {Q_k, R_k}
def unitaryMatrix(n):
  X = np.random.rand(n, n) / np.sqrt(2)
  Q, R = qr(X)
  R = np.diag(np.diag(R) / np.abs(np.diag(R)))
  return Q @ R


def composeBlockDiagonalMatrix(D, O, N, nx):
  out = np.zeros((N*nx, N*nx))
  out[0:nx, 0:2*nx] = np.hstack((D[0], O[0]))
  for i in range(1, N-1):
    row = slice(i*nx, (i+1)*nx)
    col = slice((i-1)*nx, (i+2)*nx)
    out[row, col] = np.hstack((O[i-1].T, D[i], O[i]))
  out[-nx:, -2*nx:] = np.hstack((O[N-2].T, D[N-1]))
  return out


def formKKTSchur(A, B, Q, R, N):
  D = []
  O = []
  nx = A[0].shape[0]

  D.append(inv(Q[0]))
  for i in range(N-1):
    Di = A[i] @ inv(Q[i]) @ A[i].T + B[i] @ inv(R[i]) @ B[i].T + inv(Q[i+1])
    Oi = -A[i] @ inv(Q[i])
    O.append(Oi.T)
    D.append(Di)

  S = composeBlockDiagonalMatrix(D, O, N, nx)
  return D, O, S


def writeBlkTriDiagSymMatrixToFile(D, O, N, nx, filename):
  out = np.zeros((N*3, nx*nx))
  out[1, :] = D[0].reshape(-1)
  out[2, :] = O[0].T.reshape(-1)

  for i in range(1, N-1):
    offset = i*3
    out[offset, :] = O[i-1].reshape(-1)
    out[offset+1, :] = D[i].reshape(-1)
    out[offset+2, :] = O[i].T.reshape(-1)

  out[-3, :] = O[N-2].reshape(-1)
  out[-2, :] = D[N-1].reshape(-1)

  np.savetxt(filename, out)

def compose_block_tridiag(D_blocks, O_blocks, N, nx):
    out = np.zeros((N * nx, N * nx), dtype=float)
    out[0:nx, 0:2*nx] = np.hstack((D_blocks[0], O_blocks[0]))
    for i in range(1, N-1):
        r = slice(i*nx, (i+1)*nx)
        c = slice((i-1)*nx, (i+2)*nx)
        out[r, c] = np.hstack((O_blocks[i-1].T, D_blocks[i], O_blocks[i]))
    out[-nx:, -2*nx:] = np.hstack((O_blocks[N-2].T, D_blocks[N-1]))
    return out

def form_preconditioner_P(D, O, N, nx):
    D_P = [None] * N
    O_P = [None] * (N - 1)

    for i in range(N - 1):
        Di_inv = np.linalg.inv(D[i])
        Dip1_inv = np.linalg.inv(D[i + 1])
        D_P[i] = Di_inv
        O_P[i] = -Di_inv @ O[i] @ Dip1_inv

    D_P[N - 1] = np.linalg.inv(D[N - 1])
    P = compose_block_tridiag(D_P, O_P, N, nx)
    return D_P, O_P, P

def composeBlockPentDiagMatrix(D, O_up1, O_up2, O_down1, O_down2, N, nx):
  out = np.zeros((N * nx, N * nx))

  out[0:nx, 0:3*nx] = np.hstack((D[0], O_up1[0], O_up2[0]))
  out[nx:2*nx, 0:4*nx] = np.hstack((O_down1[0], D[1], O_up1[1], O_up2[1]))

  for i in range(3, N-1):
      out[(i-1)*nx:i*nx, (i-3)*nx:(i+2)*nx] = np.hstack((
          O_down2[i-3], O_down1[i-2], D[i-1], O_up1[i-1], O_up2[i-1]
      ))

  out[(N-2)*nx:(N-1)*nx, (N-4)*nx:] = np.hstack((
      O_down2[N-4], O_down1[N-3], D[N-2], O_up1[N-2]
  ))

  out[(N-1)*nx:N*nx, (N-3)*nx:] = np.hstack((
      O_down2[N-3], O_down1[N-2], D[N-1]
  ))

  return out

def form_poly_preconditioner_H(D, O, N, nx):
    _, O_add, _ = form_preconditioner_P(D, O, N, nx)

    O_up1 = [np.zeros((nx, nx), dtype=float) for _ in range(N - 1)]
    O_down1 = [np.zeros((nx, nx), dtype=float) for _ in range(N - 1)]

    O_up2 = [None] * (N - 2)
    O_down2 = [None] * (N - 2)
    D_H = [None] * N

    D_H[0] = -O_add[0] @ O[0].T
    D_H[N - 1] = -O_add[N - 2].T @ O[N - 2]

    for i in range(1, N - 1):
        D_H[i] = -O_add[i] @ O[i].T - O_add[i - 1].T @ O[i - 1]
        O_up2[i - 1] = -O_add[i - 1] @ O[i]
        O_down2[i - 1] = -O_add[i].T @ O[i - 1].T

    H = composeBlockPentDiagMatrix(D_H, O_up1, O_up2, O_down1, O_down2, N, nx)
    return D_H, O_up2, O_down2, H

def write_blk_pentadiag_to_file(D, O_up2, O_down2, N, nx, filename):
    out = np.zeros((N * 3, nx * nx))

    out[1, :] = D[0].T.reshape(-1)
    out[2, :] = O_up2[0].T.reshape(-1)

    out[4, :] = D[1].T.reshape(-1)
    out[5, :] = O_up2[1].T.reshape(-1)

    for i in range(2, N-2):
        offset = i * 3
        out[offset + 0, :] = O_down2[i-2].T.reshape(-1)
        out[offset + 1, :] = D[i].T.reshape(-1)
        out[offset + 2, :] = O_up2[i].T.reshape(-1)

    out[-6, :] = O_down2[N-4].T.reshape(-1)
    out[-5, :] = D[N-2].T.reshape(-1)

    out[-3, :] = O_down2[N-3].T.reshape(-1)
    out[-2, :] = D[N-1].T.reshape(-1)

    np.savetxt(filename, out)

if __name__ == "__main__":
  # nx = int(sys.argv[1])
  # N = int(sys.argv[2])
  # np.random.seed(0)
  nx = 12
  N = 50
  nu = 1
  Nnx = N*nx

  if not os.path.exists("data"):
    os.mkdir("data")
  else:
    for f in os.listdir("data"):
      if f.startswith("I_H"):
        os.remove(os.path.join("data", f))

  # generate random {A_k, B_k}
  A = [np.random.rand(nx, nx) for _ in range(N)]
  B = [np.random.rand(nx, nu) for _ in range(N)]
  h_gamma = np.random.rand(Nnx)

  Q = []
  R = []
  for i in range(N-1):
    T = unitaryMatrix(nx)
    Q.append(T @ np.diag(np.random.rand(nx)) @ T.T)
    R.append(np.diag(np.random.rand(nu)))
  T = unitaryMatrix(nx)
  Q.append(T @ np.diag(np.random.rand(nx)) @ T.T)

#   # ----------------
#   # A matrices (N-1 = 3)
#   # ----------------
#   A = [
#       np.array([
#           [0.609857169290216, 0.696432989006095, 0.007820293569335, 0.531209293582439],
#           [0.059403296858277, 0.125332181109180, 0.423109385164167, 0.108817938273045],
#           [0.315811438338866, 0.130151450389424, 0.655573174937914, 0.631766373528489],
#           [0.772722130862935, 0.092352338719202, 0.722922524692024, 0.126499865329303]
#       ]),
#       np.array([
#           [0.196248922256955, 0.251041846015736, 0.184433667757653, 0.706715217696931],
#           [0.317479775149435, 0.892922405285977, 0.212030842532321, 0.557788966754876],
#           [0.316428999146291, 0.703223224556291, 0.077346808112677, 0.313428989936591],
#           [0.217563309422821, 0.555737942719387, 0.913800410779568, 0.166203562902151]
#       ]),
#       np.array([
#           [0.396799318633144, 0.982835201393951, 0.381345204444472, 0.350776744885893],
#           [0.073994769576938, 0.402183985222485, 0.161133971849361, 0.685535708747537],
#           [0.684096066962009, 0.620671947199578, 0.758112431327419, 0.294148633767850],
#           [0.402388332696162, 0.154369805479272, 0.871111121915389, 0.530629303856886]
#       ]),

#       np.array([
#           [0.452592541569324,	0.742545365701939,	0.024434016050374,	0.956935924070684],
#           [0.422645653220462,	0.424334783625691,	0.290185265130727,	0.935730872784880],
#           [0.359606317972236,	0.429355788576205,	0.317520582899226,	0.457886333854367],
#           [0.558319199869297,	0.124872758719813,	0.653690133966475,	0.240478396832085]
#       ])
#   ]

#   # ----------------
#   # B matrices (N-1 = 3)
#   # ----------------
#   B = [
#       np.array([[0.1343], [0.0986], [0.1420], [0.1683]]),
#       np.array([[0.6225], [0.9879], [0.1704], [0.2578]]),
#       np.array([[0.8324], [0.5975], [0.3353], [0.2992]]),
#       np.array([[0.7639], [0.7593], [0.7406], [0.7437]])
#   ]

#   # ----------------
#   # Q matrices (N = 4)
#   # ----------------
#   Q = [
#       np.array([
#           [0.484526123347466, -0.088433602380418, -0.101109343386961,  0.281219253636585],
#           [-0.088433602380418,  0.317986054415282, -0.338060632013289, -0.121744568177382],
#           [-0.101109343386961, -0.338060632013289,  0.568790897667888, -0.025649602071676],
#           [0.281219253636585, -0.121744568177382, -0.025649602071676,  0.343269683674888]
#       ]),
#       np.array([
#           [0.610116134916631, -0.156927071532446,  0.214357706226574,  0.111932920751231],
#           [-0.156927071532446,  0.597096549460243,  0.159871082817675,  0.173407316128773],
#           [0.214357706226574,  0.159871082817675,  0.259715965669101,  0.094279581106786],
#           [0.111932920751231,  0.173407316128773,  0.094279581106786,  0.345203824095450]
#       ]),
#       np.array([
#           [0.519005269075272, -0.098229639231006, -0.116728964827208, -0.013464630676103],
#           [-0.098229639231006,  0.698015082620436, -0.149862106784363, -0.222986660340229],
#           [-0.116728964827208, -0.149862106784363,  0.542828596083302,  0.258541502688398],
#           [-0.013464630676103, -0.222986660340229,  0.258541502688398,  0.468850547658228]
#       ]),
#       np.array([
#           [0.266722052075946, -0.172410679527037, -0.147552887974793, -0.097787372994429],
#           [-0.172410679527037,  0.370942076109032,  0.164770564470377,  0.053007168712603],
#           [-0.147552887974793,  0.164770564470377,  0.160612639089691,  0.075688243779441],
#           [-0.097787372994429,  0.053007168712603,  0.075688243779441,  0.155008125775314]
#       ])
#   ]

#   # ----------------
#   # R matrices (N-1 = 3)
#   # ----------------
#   R = [
#       np.array([[0.5250]]),
#       np.array([[0.6712]]),
#       np.array([[0.6050]])
# ]


  D, O, S = formKKTSchur(A, B, Q, R, N)
  writeBlkTriDiagSymMatrixToFile(D, O, N, nx, "./data/h_S.txt")

  D_P, O_P, P = form_preconditioner_P(D, O, N, nx)
  writeBlkTriDiagSymMatrixToFile(D_P, O_P, N, nx, "./data/P.txt") 

  D_H, O_up2, O_down2, H = form_poly_preconditioner_H(D, O, N, nx)
  write_blk_pentadiag_to_file(D_H, O_up2, O_down2, N, nx, "./data/H.txt")

  np.savetxt("./data/S.txt", S)
  np.savetxt("./data/h_gamma.txt", h_gamma)

