## Purpose

This file gives concise, repository-specific directions for AI coding agents (Copilot-style) to be immediately productive in GBD-PCG.
Focus areas: where core logic lives, build & run workflows, project-specific conventions (compile-time macros, data layout), and useful files to inspect when changing behavior.

## Big picture (what this project is)

- Project: a GPU-accelerated preconditioned conjugate gradient (PCG) solver for block‑tridiagonal linear systems used in optimal control (see top-level `README.md`).
- Major components:
  - `src/` — core CUDA implementations that build into `libgpupcg.a` (see top-level `Makefile`).
  - `include/` — public headers, type definitions and kernel interfaces (.cuh files).
  - `GLASS/` — low-level GPU linear-algebra building blocks (gemm, chol, norms). Use this for performance-sensitive kernels.
  - `Benchmark/` — benchmarking drivers and Python scripts that run comparisons and generate JSON/plots.
  - `examples/` — sample CUDA drivers and input data showing typical invocation patterns.

## Quick build & run (common workflows)

- Build the core library (uses `nvcc`):
  - From repo root: `make` — this compiles `src/*.cu` to `libgpupcg.a` (top-level `Makefile`).
- Compile a standalone CUDA example (from README):
  - `nvcc -I../include -I../GLASS -DKNOT_POINTS=3 -DSTATE_SIZE=2 pcg_solve.cu -o pcg.exe`
  - Important: `KNOT_POINTS` and `STATE_SIZE` are compile-time macros required by the code; they must match any values passed at the API level.
- Benchmarks (Python orchestrator + compiled CUDA binaries):
  - Install python deps: `pip install numpy scipy matplotlib`.
  - Run: `python3 Benchmark/benchmark.py` — GPU binaries live under `Benchmark/*` (each subfolder typically has its own `makefile`).
- GLASS tests: look in `GLASS/GTests/` — tests use GoogleTest (`-lgtest`). Build/run via the GTests Makefile.

## Project-specific conventions and patterns

- Matrix format: block-tridiagonal / compressed formats. See `README.md` and `include/*` for `csr_t` / `cbtd_t` types. When modifying storage/layout, update all callers (benchmarks, examples).
- Compile-time constants: `KNOT_POINTS` and `STATE_SIZE` are used as `-D` macros in many build commands and are expected by kernels as compile-time constants. Changes to these often require recompilation of both library and examples.
- Directory responsibilities:
  - `include/` contains API contracts and kernel prototypes — change carefully (ABI-sensitive).
  - `GLASS/` implements reusable GPU building blocks — preferred place to add optimized kernels.
  - `Benchmark/` includes both CPU and GPU implementations; the Python harness reads/writes JSON in `Benchmark/data/` and plots to `Benchmark/plots/`.

## Integration points & external dependencies

- CUDA toolkit (nvcc) — README specifies CUDA 11.0+; some bench scripts mention 11.8+. Ensure GPU drivers/toolkit are compatible with your hardware.
- Python (>=3.8) with numpy/scipy/matplotlib for benchmarks and plotting.
- GoogleTest for GLASS unit tests.

## Files to inspect first when making changes

- `Makefile` (repo root) — builds `libgpupcg.a` from `src/*.cu`.
- `src/interface.cu`, `src/pcg.cu` — core PCG logic and public entry points.
- `include/pcg.cuh`, `include/precondition.cuh`, `include/utils.cuh` — data layouts, config structs, preconditioner hooks.
- `GLASS/src/` — low-level kernels (gemm, chol, inv) and `GLASS/GTests/` for test patterns.
- `Benchmark/benchmark.py` and `Benchmark/` subfolders — how the project runs experiments and consumes produced JSON data.

## Editing guidance & small examples

- Adding a new CUDA kernel:
  1. Implement kernel in `GLASS/src/` or `src/` (depending on scope).
  2. Add header to `include/` if it becomes public API.
  3. Update the appropriate `Makefile` to compile the new `.cu` file into the library/binary.
  4. Update any benchmarks/examples that rely on the kernel.

- Typical compile example (root-level lib):
  - `make` → produces `libgpupcg.a` from `src/*.cu`.

## What to avoid / gotchas discovered in the codebase

- Don’t assume runtime-configurable state sizes — many kernels expect `STATE_SIZE`/`KNOT_POINTS` as compile-time macros and the code often duplicates those values in both API and build-time defines.
- Modifying the block-tridiagonal storage requires sweeping changes: headers, kernel index math, and callers in `Benchmark/` and `examples/`.

## Where to leave hints for future AI agents (and humans)

- Prefer small, localized README snippets near non-obvious components (e.g., `include/` headers, `GLASS/` subfolders) describing expected invariants (row-major vs column-major, block sizes, required CUDA compute capability).

## Feedback

If anything above is unclear or you want more detail about a specific area (e.g., preconditioner design in `include/precondition.cuh`, or how `Benchmark/benchmark.py` orchestrates runs), tell me which area and I will expand the instructions and add concrete code examples.
