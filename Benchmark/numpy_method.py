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
  if len(sys.argv) != 4:
    print("Usage: python3 scipy.py <state> <horizon> <PRINT_ERROR>")
    sys.exit(1)

  nx = int(sys.argv[1])
  N = int(sys.argv[2])
  PRINT_ERROR = sys.argv[3].lower() in ("1", "true", "True")
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
  print(f"{exec_time_ms:.6f}")

  if PRINT_ERROR :
    r = A @ x - b
    res_norm = np.linalg.norm(r)
    print(res_norm)

if __name__ == "__main__":
    main()

