import subprocess
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import json

def run_cmd(cmd):
  """Execute command Shell and return the execution time as float."""
  result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
  return float(result.stdout.strip())

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

def write_data(filtered_results, avg, x_axis_values, methods, benchmark_name="default", base_dir="data"):
    """
    Save filtered and averaged results in a subdirectory for each benchmark.

    Parameters
    ----------
    filtered_results : list[list[list[float]]]
        filtered_results[i][j] = list of runtimes (after filtering) for x_axis_values[i] and methods[j]
    avg : list[list[float]]
        avg[i][j] = average runtime for x_axis_values[i] and methods[j]
    x_axis_values : list[float]
        Values along the benchmark's varying dimension (e.g. state size or horizon)
    methods : list[str]
        Names of the tested methods
    benchmark_name : str
        Name of the benchmark (subdirectory under "data/"), e.g. "state_size" or "horizon"
    base_dir : str
        Root directory for all data (default: "data/")
    """
    # Dossier de sortie spécifique à ce benchmark
    output_dir = os.path.join(base_dir, benchmark_name)
    os.makedirs(output_dir, exist_ok=True)

    n_values = len(x_axis_values)
    n_methods = len(methods)

    for method_idx, method in enumerate(methods):
        method_data = {
            "x_axis_values": x_axis_values,
            "method": method,
            "filtered_results": [
                filtered_results[val_idx][method_idx]
                for val_idx in range(n_values)
            ],
            "avg": [
                avg[val_idx][method_idx]
                for val_idx in range(n_values)
            ],
        }

        file_path = os.path.join(output_dir, f"{method}.json")
        with open(file_path, "w") as f:
            json.dump(method_data, f, indent=2)

        print(f"💾 Saved data for method '{method}' in {file_path}")


def read_data(benchmark_name, base_dir="data"):
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

    print(f"📂 Loaded benchmark '{benchmark_name}' ({n_methods} methods, {n_values} points)")
    return filtered_results, avg, x_axis_values, methods


def compute_run(num_runs, state_sizes, methods, method_paths, knot_points):
  """
  Compile and execute each requested method over multiple runs, model sizes, 
  and knot-point configurations. Handles Python scripts as well as C++/CUDA 
  sources that require on-the-fly compilation. Execution times for every 
  (run, size, knot point, method) combination are measured and stored.

  Parameters
  ----------
  num_runs : int
      Number of repeated benchmark executions.
  state_sizes : list[int]
      List of model state dimensions to test.
  methods : list[str]
      Names of the benchmarking methods to execute.
  method_paths : dict[str, str]
      Mapping from method name to its Python/C++/CUDA entry file.
  knot_points : list[int]
      List of time-discretization knot-point counts to benchmark.
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
  results = [[[0.0 for _ in methods] for _ in range(num_points)] for _ in range(num_runs)]

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
            f"-DSTATE_SIZE={state_size} -DKNOT_POINTS={kp} "
            f"-I../../include"
            f"-I../../GLASS"
            f"-I../include"
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
        exec_time_ms = run_cmd(cmd)

        results[run_idx][point_idx][method_idx] = exec_time_ms
        print(f"⏱️  Time = {exec_time_ms:.3f} ms\n")

  # -------- Final Cleanup --------
  print("🧹 Cleaning executables...")
  for exe in set(executables_to_cleanup):
      subprocess.run(f"rm -f {exe}", shell=True)
  print("✔️ Cleanup complete.")

  return results




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

  # results[run][model_size][method]
  results = compute_run(nbr_run, model_states_sizes, methods, method_paths, model_knot_point)

  # Filter and average
  filtered_results, avg = eliminate_outliers(results)

  write_data(filtered_results, avg, sizes, methods, file_name)
  data_filtred_result, data_avg, data_size, data_methods = read_data(file_name)

  # Plot results
  plot_filtered_results(data_filtred_result,
                        data_avg, data_size, 
                        data_methods,
                        file_name + ".png",
                        x_label=x_label,
                        y_label=y_label,
                        title=title_plot)
   
def benchmark():

  nbr_run = 5
  # methods = ["numpy1", "eigen1", "pcg_no_gpu1", "pcg_no_precond1", "pcg_precond1"]
  methods = ["numpy1", "pcg_no_gpu1", "pcg_no_precond1", "pcg_precond1"]
  method_paths = {
    "numpy": "linlag.py",
    # "eigen": "./eigen/eigen.cpp",
    "pcg_no_gpu": "./CG_no_GPU/benchmark_CG_no_GPU.cu",
    "pcg_no_precond" : "./CG_no_precond/CG_no_precond.cu",
    "pcg_precond" : "./CG_precond/CG_precond.cu"
  }

  # first run states_sizes
  model_knot_point = [50]
  model_states_sizes = [7*i for i in range(1, 7)]
  save_data_plot(nbr_run,
                 model_states_sizes,
                 methods,
                 method_paths,
                 model_knot_point,
                 "state",
                 "Benchmark of the number of states with horizon = 50",
                 "increase the number of states")

  # second run knot_point
  model_knot_point = [15*i for i in range(1, 7)]
  model_states_sizes = [30]
  save_data_plot(nbr_run,
                 model_states_sizes,
                 methods,
                 method_paths,
                 model_knot_point,
                 "horizon",
                 "Benchmark of the horizon with state size = 30",
                 "increase the horizon")


def benchmark_only_plot() :
  data_filtred_result, data_avg, data_size, data_methods = read_data("state")
  plot_filtered_results(data_filtred_result,
                        data_avg, data_size,
                        data_methods,
                        "state" + ".png",
                        x_label="increase the number of states, horizon = 50",
                        title="Benchmark of the number of states with horizon = 50")

  data_filtred_result, data_avg, data_size, data_methods = read_data("horizon")
  plot_filtered_results(data_filtred_result, 
                        data_avg, 
                        data_size, 
                        data_methods, 
                        "horizon" + ".png", 
                        x_label="increase the horizon, state size = 30", 
                        title="Benchmark of the horizon with state size = 30")



if __name__ == "__main__":
  # benchmark()
  benchmark_only_plot()
