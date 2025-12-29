# Benchmarking Solvers for Block-Tridiagonal SPD Linear Systems

## Overview

This repository presents a systematic benchmark of numerical methods for solving large-scale **symmetric positive definite (SPD) linear systems** of the form:

```
A x = b
```

where the matrix `A` has a **block-tridiagonal structure**.
Such systems commonly arise in optimal control, trajectory optimization, and model predictive control (MPC). `A` has this form : 

```
| D1   O1^T   0      ...       0     |
| O1    D2    O2^T   ...       0     |
| 0     O2    D3     ...       0     |
| ...   ...   ...     ...     On-1^T |
| 0     0     0      On-1     Dn     |
```

Where:

* `Dk` are diagonal blocks of size `(state × state)`
* `Ok` are off-diagonal coupling blocks of size `(state × state)`
* all `Dk` are symmetric positive definite

The right-hand side vector `b` is generated consistently with this structure.

The benchmark compares CPU-based and GPU-based solvers.

---

## 1. Software and Hardware Requirements

### Hardware
* NVIDIA GPU
  * Compute Capability ≥ 7.0
  * Large shared-memory support
* x86_64 CPU

### Software
* Linux (tested on Ubuntu)
* `nvcc` and CUDA Toolkit ≥ 12.0
* `g++` (C++17)
* Python ≥ 3.8

### Python Dependencies

```txt
numpy>=1.23
scipy>=1.9
matplotlib>=3.6
```

Install with:

```bash
pip install -r requirements.txt
```

---


## 2. Evaluated Methods

### CPU Solvers

* **NumPy**
  * Sparse Chollesky solver using `numpy.linalg.solve`
* **Eigen**
  * Sparse Cholesky solver using `Eigen`

### GPU Direct Solver

* **cuDSS (Cholesky)**

  * GPU-accelerated sparse Cholesky factorization

### GPU Iterative Solvers (PCG)

* **Yang – PCG without preconditioning**
* **Yang – PCG with block preconditioner**
* **Yang – Optimized preconditioned PCG**
  * try my own improvment
* **Gato – PCG with block preconditioner**

All PCG solvers solve the same linear system

---

## 3. Benchmark Suite

The benchmark consists of **ten experiments** and we run it each 50 times for avoid the extreme result:

1. State size vs GPU kernel execution time
2. Horizon size vs GPU kernel execution time
3. State size vs total execution time (host + device)
4. Horizon size vs total execution time (host + device)
5. State size vs number of PCG iterations
6. Horizon size vs number of PCG iterations
7. Iterations vs L2 error (state = 21, horizon = 40)
8. Iterations vs L2 error (state = 30, horizon = 90)
9. Kernel time vs error (state = 21, horizon = 40)
10. Kernel time vs error (state = 30, horizon = 90)

Each experiment is repeated multiple times, and statistical outliers are filtered before averaging.

---

## 4. Running the Benchmarks

Run the complete benchmark suite with:

```bash
python3 benchmark.py
```

This script:

1. Generates SPD block-tridiagonal matrices
2. Compiles each solver with problem-specific parameters
3. Executes all benchmarks
4. Stores results in `datas/`
5. Generates plots in `plots/`

---

## 5. Report

A detailed discussion of methodology and results is available in the accompanying report:

```
[Link to report]
```

---
