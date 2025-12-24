import numpy as np
import sys
import time
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

def readMatrifromFile(filename, dim):
  data = np.loadtxt(filename)
  if data.size != dim * dim:
    raise ValueError("Matrix dim is incorrect")
  return data.reshape((dim, dim))

def readVectorfromFile(filename, dim):
  data = np.loadtxt(filename)
  if data.size != dim:
    raise ValueError("Vector dim is incorrect")
  return data.reshape((dim,))

def main():
  # -------- setup problem dimension  ---------
  nx = int(sys.argv[1])
  N = int(sys.argv[2])
  STATExKERNEL = sys.argv[3].lower() in ("1", "true")
  KNOTxKERNEL = sys.argv[4].lower() in ("1", "true")
  STATExCOMPUTER = sys.argv[5].lower() in ("1", "true")
  KNOTxCOMPUTER = sys.argv[6].lower() in ("1", "true")
  KERNELxERROR = sys.argv[7].lower() in ("1", "true")
  Nnx = N * nx

  # -------- data reading  ---------
  S = readMatrifromFile(f"./include/data/S.txt", Nnx)
  b = readVectorfromFile(f"./include/data/h_gamma.txt", Nnx)

  A = csc_matrix(S)

  # -------- Solve  ---------
  start = time.perf_counter()
  x = spsolve(A, b)
  end = time.perf_counter()

  # -------- print  ---------
  exec_time_ms = (end - start) * 1000
  if STATExKERNEL or STATExCOMPUTER:
    print(f"{nx}")
    print(f"{exec_time_ms:.6f}")
  if KNOTxKERNEL or KNOTxCOMPUTER:
    print(f"{N}")
    print(f"{exec_time_ms:.6f}")

  if KERNELxERROR:
    r = A @ x - b
    res_norm = np.linalg.norm(r)
    print(f"{exec_time_ms:.6f}")
    print(res_norm)

if __name__ == "__main__":
    main()

