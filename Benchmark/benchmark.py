import subprocess
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import json


def run_cmd(cmd):
    """
    Execute shell command and return execution times as floats.

    Expected stdout formats:
      - "12.34"                 -> total = 12.34, kernel = 12.34
      - "12.34\\n5.67"          -> total = 12.34, kernel = 5.67
    """
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    stdout = result.stdout.strip()
    if not stdout:
        raise RuntimeError(f"Command produced no output: {cmd}")

    lines = stdout.splitlines()
    # Première ligne = temps total
    total_time = float(lines[0].strip())

    # Deuxième ligne (si présente) = temps kernel
    if len(lines) >= 2 and lines[1].strip() != "":
        kernel_time = float(lines[1].strip())
    else:
        # Si une seule valeur : on considère kernel_time = total_time
        kernel_time = total_time

    return total_time, kernel_time


def gauss_filter(data):
    """Filter extreme values according to a normal distribution (95% confidence interval)."""
    if len(data) <= 2:
        return [], None
    mu, sigma = np.mean(data), np.std(data)
    lower_bound = norm.ppf(0.025, loc=mu, scale=sigma)
    upper_bound = norm.ppf(0.975, loc=mu, scale=sigma)
    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
    mean_filtered = np.mean(filtered_data) if filtered_data else None
    return filtered_data, mean_filtered


def eliminate_outliers(results):
    """
    results: [nbr_run][nbr_model_sizes][nbr_methods]
    Returns:
      filtered_results[model_size][method] = filtered values
      avg[model_size][method] = filtered mean
    """
    nbr_run = len(results)
    nbr_model_sizes = len(results[0])
    nbr_methods = len(results[0][0])

    filtered_results = [[[] for _ in range(nbr_methods)] for _ in range(nbr_model_sizes)]
    avg = [[None for _ in range(nbr_methods)] for _ in range(nbr_model_sizes)]

    for i in range(nbr_model_sizes):  # size model
        for j in range(nbr_methods):  # methods
            all_runs = [results[k][i][j] for k in range(nbr_run)]
            filtered_data, mean = gauss_filter(all_runs)
            filtered_results[i][j] = filtered_data
            avg[i][j] = mean
    return filtered_results, avg


def plot_filtered_results(filtered_results, avg, model_sizes, method_names,
                          file_name,
                          save_path=None,
                          x_label="Model size (states)",
                          y_label="Execution time [ms]",
                          title="Benchmark Results",
                          ):
    """
    Affiche et/ou sauvegarde les résultats filtrés.
    Si save_path est fourni, le plot est enregistré dans ce fichier (PNG, PDF, etc.)
    """

    colors = plt.cm.tab10.colors
    num_methods = len(method_names)
    num_sizes = len(model_sizes)

    plt.figure(figsize=(12, 6))

    # Moyennes
    for j in range(num_methods):
        avg_values = [avg[i][j] for i in range(num_sizes) if avg[i][j] is not None]
        valid_sizes = [model_sizes[i] for i in range(num_sizes) if avg[i][j] is not None]

        if not avg_values:
            continue

        plt.plot(valid_sizes, avg_values, label=method_names[j],
                 color=colors[j % len(colors)], marker='s', linewidth=2)

        # Points filtrés individuels
        for i in range(num_sizes):
            for val in filtered_results[i][j]:
                plt.scatter(model_sizes[i], val,
                            color=colors[j % len(colors)], marker='x', alpha=0.6)

    plt.xticks(model_sizes, model_sizes)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(title='Method')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()

    if save_path is None:
        project_dir = os.path.dirname(__file__)
        plots_dir = os.path.join(project_dir, "plots")
        save_path = os.path.join(plots_dir, file_name)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"✅ Figure sauvegardée dans : {save_path}")
    else:
        plt.show()

    plt.close()


def write_data(filtered_results_total, avg_total, x_axis_values, methods,
               benchmark_name="default", base_dir="data",
               filtered_results_kernel=None, avg_kernel=None):
    """
    Save filtered and averaged results in a subdirectory for each benchmark.

    Si filtered_results_kernel / avg_kernel sont fournis, on écrit un JSON étendu :
      - filtered_results_total
      - filtered_results_kernel
      - avg_total
      - avg_kernel

    Sinon, on utilise l'ancien format (filtered_results / avg) pour compatibilité.
    """
    # Dossier de sortie spécifique à ce benchmark
    output_dir = os.path.join(base_dir, benchmark_name)
    os.makedirs(output_dir, exist_ok=True)

    n_values = len(x_axis_values)
    n_methods = len(methods)

    for method_idx, method in enumerate(methods):
        if filtered_results_kernel is None or avg_kernel is None:
            # Ancien format (compatibilité)
            method_data = {
                "x_axis_values": x_axis_values,
                "method": method,
                "filtered_results": [
                    filtered_results_total[val_idx][method_idx]
                    for val_idx in range(n_values)
                ],
                "avg": [
                    avg_total[val_idx][method_idx]
                    for val_idx in range(n_values)
                ],
            }
        else:
            # Nouveau format avec total + kernel
            method_data = {
                "x_axis_values": x_axis_values,
                "method": method,
                "filtered_results_total": [
                    filtered_results_total[val_idx][method_idx]
                    for val_idx in range(n_values)
                ],
                "filtered_results_kernel": [
                    filtered_results_kernel[val_idx][method_idx]
                    for val_idx in range(n_values)
                ],
                "avg_total": [
                    avg_total[val_idx][method_idx]
                    for val_idx in range(n_values)
                ],
                "avg_kernel": [
                    avg_kernel[val_idx][method_idx]
                    for val_idx in range(n_values)
                ],
            }

        file_path = os.path.join(output_dir, f"{method}.json")
        with open(file_path, "w") as f:
            json.dump(method_data, f, indent=2)

        print(f"💾 Saved data for method '{method}' in {file_path}")


def read_data(benchmark_name, base_dir="data", metric="total"):
    """
    Load JSON files from a specific benchmark subdirectory and reconstruct:
      - filtered_results[i][j]
      - avg[i][j]
      - x_axis_values
      - methods

    Parameters
    ----------
    benchmark_name : str
        Name of the benchmark subdirectory under "data/"
    base_dir : str
        Root data directory (default: "data/")
    metric : str
        "total" ou "kernel".
        - Si le fichier est au nouveau format, on lit filtered_results_total / kernel.
        - Si le fichier est à l'ancien format, on lit filtered_results / avg pour n'importe quel metric.
    """
    data_dir = os.path.join(base_dir, benchmark_name)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"❌ Directory not found: {data_dir}")

    files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    if not files:
        raise FileNotFoundError(f"❌ No JSON files found in {data_dir}")

    methods = []
    per_method_filtered = {}
    per_method_avg = {}
    x_axis_values = None

    for filename in files:
        file_path = os.path.join(data_dir, filename)
        with open(file_path, "r") as f:
            data = json.load(f)

        method = data["method"]
        methods.append(method)

        if x_axis_values is None:
            x_axis_values = data["x_axis_values"]

        # Nouveau format
        if "filtered_results_total" in data and "avg_total" in data:
            if metric == "total":
                per_method_filtered[method] = data["filtered_results_total"]
                per_method_avg[method] = data["avg_total"]
            elif metric == "kernel":
                per_method_filtered[method] = data["filtered_results_kernel"]
                per_method_avg[method] = data["avg_kernel"]
            else:
                raise ValueError(f"Unknown metric '{metric}', expected 'total' or 'kernel'.")
        # Ancien format (compatibilité)
        else:
            per_method_filtered[method] = data["filtered_results"]
            per_method_avg[method] = data["avg"]

    n_values = len(x_axis_values)
    n_methods = len(methods)

    filtered_results = [
        [per_method_filtered[methods[m]][i] for m in range(n_methods)]
        for i in range(n_values)
    ]

    avg = [
        [per_method_avg[methods[m]][i] for m in range(n_methods)]
        for i in range(n_values)
    ]

    print(f"📂 Loaded benchmark '{benchmark_name}' ({n_methods} methods, {n_values} points, metric={metric})")
    return filtered_results, avg, x_axis_values, methods


def compute_run(num_runs, state_sizes, methods, method_paths, knot_points):
    """
    Compile and execute each requested method over multiple runs, model sizes,
    and knot-point configurations. Handles Python scripts as well as C++/CUDA
    sources that require on-the-fly compilation. Execution times for every
    (run, size, knot point, method) combination are measured and stored.

    Returns
    -------
    results_total : [run][point][method] -> temps total
    results_kernel : [run][point][method] -> temps kernel
    """

    # Validate that state sizes and knot points are compatible
    if len(state_sizes) > 1 and len(knot_points) > 1 and len(state_sizes) != len(knot_points):
        raise ValueError("model_states_sizes and model_knot_points must match in length or one must have length 1.")

    # Normalize lengths
    num_points = max(len(state_sizes), len(knot_points))
    if len(state_sizes) == 1:
        state_sizes *= num_points
    if len(knot_points) == 1:
        knot_points *= num_points

    # Initialize results: [run][point][method]
    results_total = [[[0.0 for _ in methods] for _ in range(num_points)] for _ in range(num_runs)]
    results_kernel = [[[0.0 for _ in methods] for _ in range(num_points)] for _ in range(num_runs)]

    executables_to_cleanup = []  # Keep track of executables to delete at the end

    # -------- New: Pre-build commands per point/method --------
    prepared_cmds = [[None for _ in methods] for _ in range(num_points)]

    for point_idx in range(num_points):
        state_size = state_sizes[point_idx]
        kp = knot_points[point_idx]

        print(f"📏 Benchmarking state_size={state_size}, kp={kp}")

        # Prepare all methods ONCE per (size, knot)
        for method_idx, method_name in enumerate(methods):

            exe_path = method_paths[method_name]
            method_dir = os.path.dirname(exe_path)
            method_base = os.path.splitext(os.path.basename(exe_path))[0]

            # Python workflow: no compile needed
            if exe_path.endswith(".py"):
                cmd = f"python3 {exe_path} {state_size} {kp}"
                prepared_cmds[point_idx][method_idx] = cmd
                continue

            # C++ / CUDA executable workflow
            cu_src = os.path.join(method_dir, f"{method_base}.cu")
            cpp_src = os.path.join(method_dir, f"{method_base}.cpp")
            exe_name = os.path.join(method_dir, f"{method_base}_{state_size}_{kp}")

            # Clean previous executable before recompilation
            subprocess.run(f"rm -f {exe_name}", shell=True)

            # Detect source type and compile
            if os.path.exists(cu_src):
                compiler = "nvcc"
                compile_cmd = (
                    f"{compiler} --compiler-options -Wall -O3 -std=c++17 "
                    f"-DTIME_EXECUTION_DOUBLE=1 "
                    f"-DSTATE_SIZE={state_size} -DKNOT_POINTS={kp} "
                    f"-I../include -I../GLASS -I./include "
                    f"{cu_src} -o {exe_name}"
                )
            elif os.path.exists(cpp_src):
                compiler = "g++"
                compile_cmd = (
                    f"{compiler} -Wall -O3 -std=c++17 "
                    f"-DTIME_EXECUTION_DOUBLE=1 "
                    f"-DSTATE_SIZE={state_size} -DKNOT_POINTS={kp} "
                    f"-I./include -I.. "
                    f"-I/usr/include/eigen3 "
                    f"{cpp_src} -o {exe_name}"
                )
            else:
                raise FileNotFoundError(f"No .cu or .cpp file found for {exe_path}")

            print(f"🔧 Compiling ({compiler}): {compile_cmd}")
            subprocess.run(compile_cmd, shell=True, check=True)

            executables_to_cleanup.append(exe_name)

            # Store final command path (no args for C++/CUDA)
            prepared_cmds[point_idx][method_idx] = exe_name

        # ---- Now run all runs for this size ----
        for run_idx in range(num_runs):
            print(f"🧪 Run {run_idx + 1}/{num_runs}")

            for method_idx, method_name in enumerate(methods):
                cmd = prepared_cmds[point_idx][method_idx]

                print(f"▶️ Execution: {cmd}")
                total_ms, kernel_ms = run_cmd(cmd)

                results_total[run_idx][point_idx][method_idx] = total_ms
                results_kernel[run_idx][point_idx][method_idx] = kernel_ms
                print(f"⏱️  Total Time = {total_ms:.3f} ms, Kernel Time = {kernel_ms:.3f} ms\n")

    # -------- Final Cleanup --------
    print("🧹 Cleaning executables...")
    for exe in set(executables_to_cleanup):
        subprocess.run(f"rm -f {exe}", shell=True)
    print("✔️ Cleanup complete.")

    return results_total, results_kernel


def save_data_plot(
        nbr_run,
        model_states_sizes,
        methods,
        method_paths,
        model_knot_point,
        file_name, title_plot,
        x_label="Model size (states)",
        y_label="Execution time [ms]"
):
    # write data and read data
    if len(model_states_sizes) == 1:
        sizes = model_knot_point
    if len(model_knot_point) == 1:
        sizes = model_states_sizes

    # results_total[run][model_size][method]
    # results_kernel[run][model_size][method]
    results_total, results_kernel = compute_run(
        nbr_run, model_states_sizes, methods, method_paths, model_knot_point
    )

    # Filter and average
    filtered_total, avg_total = eliminate_outliers(results_total)
    filtered_kernel, avg_kernel = eliminate_outliers(results_kernel)

    # Écriture JSON étendu (total + kernel)
    write_data(filtered_total, avg_total, sizes, methods, file_name,
               filtered_results_kernel=filtered_kernel, avg_kernel=avg_kernel)

    # Plot temps total (fichier comme avant : <file_name>.png)
    data_filtered_total, data_avg_total, data_size, data_methods = read_data(file_name, metric="total")
    plot_filtered_results(
        data_filtered_total,
        data_avg_total, data_size,
        data_methods,
        file_name + ".png",
        x_label=x_label,
        y_label=y_label,
        title=title_plot + " (total time)"
    )

    # Plot temps kernel (nouveau : <file_name>_kernel.png)
    data_filtered_kernel, data_avg_kernel, data_size_k, data_methods_k = read_data(file_name, metric="kernel")
    plot_filtered_results(
        data_filtered_kernel,
        data_avg_kernel, data_size_k,
        data_methods_k,
        file_name + "_kernel.png",
        x_label=x_label,
        y_label=y_label,
        title=title_plot + " (kernel time)"
    )


def benchmark():
    nbr_run = 5
    # ATTENTION : vérifier que les clés de 'methods' correspondent bien
    # aux clés de 'method_paths' dans ton repo final.
    # Ici je laisse comme dans ton exemple original.
    # methods = ["numpy", "eigen", "pcg_no_gpu", "pcg_no_precond", "pcg_precond"]
    methods = ["numpy", "eigen", "pcg_no_precond", "pcg_precond"]
    method_paths = {
        "numpy": "linlag.py",
        "eigen": "./eigen/eigen.cpp",
        # "pcg_no_gpu": "./CG_no_GPU/CG_no_GPU.cpp",
        "pcg_no_precond": "./CG_no_precond/CG_no_precond.cu",
        "pcg_precond": "./CG_precond/CG_precond.cu"
    }

    # first run states_sizes
    model_knot_point = [50]
    model_states_sizes = [7 * i for i in range(1, 7)]
    save_data_plot(nbr_run,
                   model_states_sizes,
                   methods,
                   method_paths,
                   model_knot_point,
                   "state",
                   "Benchmark of the number of states with horizon = 50",
                   "increase the number of states")

    # second run knot_point
    model_knot_point = [15 * i for i in range(1, 7)]
    model_states_sizes = [30]
    save_data_plot(nbr_run,
                   model_states_sizes,
                   methods,
                   method_paths,
                   model_knot_point,
                   "horizon",
                   "Benchmark of the horizon with state size = 30",
                   "increase the horizon")


def benchmark_only_plot():
    # Plots à partir des données JSON déjà enregistrées

    # ---- Benchmark "state" ----
    # Temps total
    data_filtred_result, data_avg, data_size, data_methods = read_data("state", metric="total")
    plot_filtered_results(
        data_filtred_result,
        data_avg, data_size,
        data_methods,
        "state" + ".png",
        x_label="increase the number of states, horizon = 50",
        title="Benchmark of the number of states with horizon = 50 (total time)"
    )

    # Temps kernel
    data_filtred_result_k, data_avg_k, data_size_k, data_methods_k = read_data("state", metric="kernel")
    plot_filtered_results(
        data_filtred_result_k,
        data_avg_k, data_size_k,
        data_methods_k,
        "state_kernel" + ".png",
        x_label="increase the number of states, horizon = 50",
        title="Benchmark of the number of states with horizon = 50 (kernel time)"
    )

    # ---- Benchmark "horizon" ----
    # Temps total
    data_filtred_result, data_avg, data_size, data_methods = read_data("horizon", metric="total")
    plot_filtered_results(
        data_filtred_result,
        data_avg,
        data_size,
        data_methods,
        "horizon" + ".png",
        x_label="increase the horizon, state size = 30",
        title="Benchmark of the horizon with state size = 30 (total time)"
    )

    # Temps kernel
    data_filtred_result_k, data_avg_k, data_size_k, data_methods_k = read_data("horizon", metric="kernel")
    plot_filtered_results(
        data_filtred_result_k,
        data_avg_k,
        data_size_k,
        data_methods_k,
        "horizon_kernel" + ".png",
        x_label="increase the horizon, state size = 30",
        title="Benchmark of the horizon with state size = 30 (kernel time)"
    )


if __name__ == "__main__":
    # benchmark()
    benchmark_only_plot()
