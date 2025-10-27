Parfait 😄
Voici **ton README complet**, au **format Markdown prêt à copier-coller**, 100 % dans le style que tu veux 👇

---

# 🧮 Benchmark — Solvers for Ax = b

This repository provides several methods to solve the linear system:

[
A x = b
]

and compares their **execution times** across various problem sizes and solver implementations.

## ⚙️ Implemented Methods

Each method is located in the `Benchmark/` directory and can be run individually or through the main `benchmark.py` script.

| Method              | Description                                                                                           |
| :------------------ | :---------------------------------------------------------------------------------------------------- |
| **`linlag.py`**     | Solves the system using NumPy’s built-in linear algebra solver (`numpy.linalg.solve`).                |
| **`Gauss_Jordan`**  | Solves the problem using the **Gauss–Jordan elimination** method implemented in C with pointers.      |
| **`CG_no_GPU`**     | Implements the **Conjugate Gradient** method *without preconditioner*, in pure C (CPU only).          |
| **`CG_no_precond`** | Uses **CUDA parallelism** (GPU) to accelerate the Conjugate Gradient method without a preconditioner. |
| **`CG`**            | Solves the problem on GPU using the **preconditioned Conjugate Gradient** method.                     |

## 🧩 Matrix Definition

The matrix **A** is semi-positive definite and block-tridiagonal, defined as:

A = [
  D₁      O₁           0        0        0
  O₁ᵀ     D₂      O₂           0        0
   0      O₂ᵀ     D₃      O₃           0
   0       0      O₃ᵀ     D₄      O₄
   0       0       0      O₄ᵀ     D₅
]

where **Dₖ** and **Oₖ** are square blocks in ℝⁿˣⁿ.


## 📊 Benchmark 1 — Varying Model Size

In the first benchmark, the time horizon is kept constant (**30**),
and the number of model states increases from **1 to 51** in steps of **10**.

Example plot:

```
(insert the correct image path here)
```

## 📈 Benchmark 2 — Varying Horizon Length

In the second benchmark, the number of model states is fixed (**30**),
and the time horizon increases from **1 to 101** in steps of **20**.

Example plot:

```
(insert the correct image path here)
```

## Installation

Make sure you are using **Python ≥ 3.8**.

Install the required dependencies:

```bash
pip install numpy scipy matplotlib
```

or with **conda**:

```bash
conda install numpy scipy matplotlib
```

If you want to run CUDA-based benchmarks, install **CUDA Toolkit ≥ 11.8** and ensure `nvcc` is available:

```bash
nvcc --version
```

You will also need a **CUDA-capable GPU** supporting the kernels used in `<checkPcgOccupancy>`.

Examples of compatible GPUs include:

* NVIDIA RTX 3060 / 3070 / 3080 / 3090
* NVIDIA A100
* NVIDIA Tesla V100 / T4
  *(complete this list depending on your hardware)*

## 🚀 Usage

To run the benchmark:

```bash
python3 benchmark.py
```

All benchmark data will be saved in the `data/` directory (as JSON files),
and plots will be automatically generated in the `plots/` directory.

You can later regenerate plots without rerunning simulations:

```bash
python3 replot.py
```

## 🧾 Notes

* Plots are automatically saved in the `plots/` directory.
* Benchmark data is stored in `data/` (JSON format).
* You can reload the saved data using the `read_data()` function without re-running the computations.
* The code is fully portable — relative paths are automatically handled.

---
