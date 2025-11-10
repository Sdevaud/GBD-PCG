import subprocess
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import json


def compile_all(method_paths):
  """
  Automatically compiles all CUDA directories listed in method_paths.

  Parameters
  ----------
  method_paths : dict
      Dictionary {method_name: path}
      where path can be either a Python script (.py) or a CUDA executable (.exe)
  """
  compiled = []

  for method, path in method_paths.items():
    # script python
    if path.endswith(".py"):
      print(f"🟢 Python '{method}' no compilation")
      continue

    # script CUDA
    dir_path = os.path.dirname(path)
    if not dir_path:
        print(f"⚠️ Path error {path}")
        continue

    print(f"🔧 CUDA compilation '{method}' in {dir_path} ...")

    try:
        subprocess.run(["make", "-C", dir_path], check=True)
        compiled.append(dir_path)
    except subprocess.CalledProcessError:
        print(f"❌ Error in compilation of : {dir_path}")

  if compiled:
    print(f"✅ end of compilation : {', '.join(compiled)}\n")

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
                          metric_name="Execution time [ms]",
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

    # plt.xscale('log')
    # plt.yscale('log')
    plt.xticks(model_sizes, model_sizes)
    plt.xlabel('Model size (log scale)')
    plt.ylabel(metric_name + ' (log scale)')
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

def compute_run(nbr_run, model_states_sizes, methods, method_paths, model_knot_points):
    """
    Run benchmarks over multiple state sizes and/or knot points.

    Parameters
    ----------
    nbr_run : int
        Number of repetitions for averaging.
    model_states_sizes : list[int]
        List of model state sizes (can be length 1).
    methods : list[str]
        Solver methods to benchmark.
    method_paths : dict[str, str]
        Mapping of method names to their executable/script paths.
    model_knot_points : list[int]
        List of knot points (can be length 1).

    Returns
    -------
    results : list[list[list[float]]]
        results[run][i][j] = execution time for run, (state_size, knot_point) index i, and method j.
    """
    # Vérification de compatibilité des tailles
    if len(model_states_sizes) > 1 and len(model_knot_points) > 1 and len(model_states_sizes) != len(model_knot_points):
        raise ValueError(
            "❌ model_states_sizes and model_knot_points must have the same length, "
            "or one of them must be of length 1."
        )

    # Déterminer la taille effective (celle qui varie)
    n_points = max(len(model_states_sizes), len(model_knot_points))

    # Étendre les listes pour avoir la même taille
    if len(model_states_sizes) == 1:
        model_states_sizes = model_states_sizes * n_points
    if len(model_knot_points) == 1:
        model_knot_points = model_knot_points * n_points

    # Crée la structure des résultats
    results = [[[0.0 for _ in methods] for _ in range(n_points)] for _ in range(nbr_run)]

    for run in range(nbr_run):
        print(f"🧪 Run {run+1}/{nbr_run}")
        for i in range(n_points):
            size = model_states_sizes[i]
            knot_point = model_knot_points[i]

            for j, method in enumerate(methods):
                exe_path = method_paths[method]

                # Commande selon le type (Python script ou binaire)
                if exe_path.endswith(".py"):
                    cmd = f"python3 {exe_path} {size} {knot_point}"
                else:
                    cmd = f"{exe_path} {size} {knot_point}"

                print(f"→ {cmd}")
                time_exec = run_cmd(cmd)
                results[run][i][j] = time_exec

    return results


def save_data_plot(nbr_run, model_states_sizes, methods, method_paths, model_knot_point, file_name):
  # results[run][model_size][method]
  results = compute_run(nbr_run, model_states_sizes, methods, method_paths, model_knot_point)

  # Filter and average
  filtered_results, avg = eliminate_outliers(results)

  # write data and read data
  if len(model_states_sizes) == 1:
    sizes = model_knot_point
  if len(model_knot_point) == 1:
    sizes = model_states_sizes

  write_data(filtered_results, avg, sizes, methods, file_name)
  data_filtred_result, data_avg, data_size, data_methods = read_data(file_name)

  # Plot results
  plot_filtered_results(data_filtred_result, data_avg, data_size, data_methods, file_name + ".png")
   
def benchmark():

  nbr_run = 50
  methods = ["numpy", "eigen", "pcg_no_gpu", "pcg_no_precond", "pcg_precond"]
  method_paths = {
    "numpy": "linlag.py",
    "eigen": "./Eigen/benchmark_Eigen.exe",
    "pcg_no_gpu": "./CG_no_GPU/benchmark_CG_no_GPU.exe",
    "pcg_no_precond" : "./CG_no_precond/CG_no_precond.exe",
    "pcg_precond" : "./CG_precond/CG_precond.exe"
  }

  compile_all(method_paths)

  # first run states_sizes
  model_knot_point = [50]
  model_states_sizes = [7*i for i in range(1, 6)]
  save_data_plot(nbr_run, model_states_sizes, methods, method_paths, model_knot_point, "state")

  # second run knot_point
  model_knot_point = [20*i for i in range(1, 8)]
  model_states_sizes = [30]
  save_data_plot(nbr_run, model_states_sizes, methods, method_paths, model_knot_point, "horizon")


def benchmark_only_plot() :
  data_filtred_result, data_avg, data_size, data_methods = read_data("state")
  plot_filtered_results(data_filtred_result, data_avg, data_size, data_methods, "state" + ".png")

  data_filtred_result, data_avg, data_size, data_methods = read_data("state")
  plot_filtered_results(data_filtred_result, data_avg, data_size, data_methods, "horizon" + ".png")



if __name__ == "__main__":
  benchmark()
  # benchmark_only_plot()
