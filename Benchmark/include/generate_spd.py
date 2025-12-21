import numpy as np
import os
from numpy.linalg import inv, qr
import sys

np.set_printoptions(precision=16, suppress=True)

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
  # ---------- arguments ----------
  if len(sys.argv) < 4 or len(sys.argv) > 5:
    print("Usage: python3 generate_spd.py <nx> <N> <nu> [output_path]")
    sys.exit(1)

  nx = int(sys.argv[1])
  N  = int(sys.argv[2])
  nu = int(sys.argv[3])
  Nnx = N * nx

  # ---------- output path ----------
  if len(sys.argv) == 5:
    data_path = sys.argv[4]
  else:
    data_path = "./data"

  # ---------- create / clean directory ----------
  os.makedirs(data_path, exist_ok=True)

  for f in os.listdir(data_path):
    if f.startswith("I_H"):
      os.remove(os.path.join(data_path, f))

  # ---------- generate random {A_k, B_k} ----------
  A = [np.random.rand(nx, nx) for _ in range(N)]
  B = [np.random.rand(nx, nu) for _ in range(N)]
  h_gamma = np.random.rand(Nnx)

  Q = []
  R = []
  for i in range(N - 1):
    T = unitaryMatrix(nx)
    Q.append(T @ np.diag(np.random.rand(nx)) @ T.T)
    R.append(np.diag(np.random.rand(nu)))

  T = unitaryMatrix(nx)
  Q.append(T @ np.diag(np.random.rand(nx)) @ T.T)

  # ---------- build matrices ----------
  D, O, S = formKKTSchur(A, B, Q, R, N)
  writeBlkTriDiagSymMatrixToFile(
    D, O, N, nx, os.path.join(data_path, "h_S.txt")
  )

  D_P, O_P, P = form_preconditioner_P(D, O, N, nx)
  writeBlkTriDiagSymMatrixToFile(
    D_P, O_P, N, nx, os.path.join(data_path, "P.txt")
  )

  D_H, O_up2, O_down2, H = form_poly_preconditioner_H(D, O, N, nx)
  write_blk_pentadiag_to_file(
    D_H, O_up2, O_down2, N, nx, os.path.join(data_path, "H.txt")
  )

  # ---------- save data ----------
  np.savetxt(os.path.join(data_path, "S.txt"), S)
  np.savetxt(os.path.join(data_path, "h_gamma.txt"), h_gamma)

  print(f"✅ Data generated in: {data_path}")

