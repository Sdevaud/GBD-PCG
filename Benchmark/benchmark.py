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
        save_path = os.path.join(plots_dir, "bench.png")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"✅ Figure sauvegardée dans : {save_path}")
    else:
        plt.show()

    plt.close()

def write_data(filtered_results, avg, model_states_sizes, methods, output_dir="data"):
    """
    Sauvegarde les données filtrées et moyennes dans un dossier (un fichier par méthode).
    On suppose :
      - filtered_results[i][j] = liste des temps (après filtrage) pour
        model_states_sizes[i] et methods[j]
      - avg[i][j] = moyenne (float) pour cette combinaison
    """
    os.makedirs(output_dir, exist_ok=True)

    n_sizes = len(model_states_sizes)
    n_methods = len(methods)

    for method_idx, method in enumerate(methods):
        method_data = {
            "model_states_sizes": model_states_sizes,
            "method": method,
            # pour chaque taille de modèle -> les temps filtrés (tous les runs conservés)
            "filtered_results": [
                filtered_results[size_idx][method_idx]
                for size_idx in range(n_sizes)
            ],
            # pour chaque taille de modèle -> la moyenne
            "avg": [
                avg[size_idx][method_idx]
                for size_idx in range(n_sizes)
            ],
        }

        file_path = os.path.join(output_dir, f"{method}.json")
        with open(file_path, "w") as f:
            json.dump(method_data, f, indent=2)

        print(f"💾 Données sauvegardées dans {file_path}")

def read_data(data_dir="data"):
    """
    Relit les fichiers JSON dans data_dir et reconstruit :
      filtered_results[i][j]
      avg[i][j]
      model_states_sizes
      methods
    """
    files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    if not files:
        raise FileNotFoundError(f"Aucun fichier .json trouvé dans {data_dir}")

    # On lit tout
    methods = []
    per_method_filtered = {}
    per_method_avg = {}
    model_states_sizes = None

    for filename in files:
        file_path = os.path.join(data_dir, filename)
        with open(file_path, "r") as f:
            data = json.load(f)

        method = data["method"]
        methods.append(method)

        if model_states_sizes is None:
            model_states_sizes = data["model_states_sizes"]

        per_method_filtered[method] = data["filtered_results"]
        per_method_avg[method] = data["avg"]

    # On reconstruit tableaux 2D indexés [size_idx][method_idx]
    n_sizes = len(model_states_sizes)
    n_methods = len(methods)

    # filtered_results[i][j] = liste des temps filtrés (runs restants)
    filtered_results = [
        [
            per_method_filtered[methods[m]][size_idx]
            for m in range(n_methods)
        ]
        for size_idx in range(n_sizes)
    ]

    # avg[i][j] = float
    avg = [
        [
            per_method_avg[methods[m]][size_idx]
            for m in range(n_methods)
        ]
        for size_idx in range(n_sizes)
    ]

    print(f"📂 Chargé depuis {data_dir}: {n_methods} méthodes, {n_sizes} tailles de modèle")
    return filtered_results, avg, model_states_sizes, methods

def compute_run(nbr_run, model_states_sizes, methods, method_paths, model_knot_point):
  results = [[[0.0 for _ in methods] for _ in model_states_sizes] for _ in range(nbr_run)]

  for run in range(nbr_run):
    print(f"🧪 Run {run+1}/{nbr_run}")
    for i, size in enumerate(model_states_sizes):
      for j, method in enumerate(methods):
        exe_path = method_paths[method]

        # Commande selon le type (Python script ou binaire)
        if exe_path.endswith(".py"):
          cmd = f"python3 {exe_path} {size} {model_knot_point}"
        else:
          cmd = f"{exe_path} {size} {model_knot_point}"

        print(f"→ {cmd}")
        time_exec = run_cmd(cmd)
        results[run][i][j] = time_exec

  return results

def benchmark():
  nbr_run = 50
  model_knot_point = 50
  model_states_sizes = [10*i+1 for i in range(0, 6)] # 1, 11, 21, 31, 41, 51
  #methods = ["gauss_jordan", "numpy", "gradient_no_gpu", "gpu_old", "gpu_new"]
  methods = ["numpy", "gradient_no_gpu"]
  method_paths = {
    "numpy": "linlag.py",                          # script Python
    "gradient_no_gpu": "./CG_no_GPU/benchmark_CG_no_GPU.exe" # exécutable CUDA
  }

  compile_all(method_paths)

  # results[run][model_size][method]
  results = [[[0.0 for _ in methods] for _ in model_states_sizes] for _ in range(nbr_run)]

  for run in range(nbr_run):
      print(f"🧪 Run {run+1}/{nbr_run}")
      for i, size in enumerate(model_states_sizes):
          for j, method in enumerate(methods):
              exe_path = method_paths[method]

              # Commande selon le type (Python script ou binaire)
              if exe_path.endswith(".py"):
                  cmd = f"python3 {exe_path} {size} {model_knot_point}"
              else:
                  cmd = f"{exe_path} {size} {model_knot_point}"

              print(f"→ {cmd}")
              time_exec = run_cmd(cmd)
              results[run][i][j] = time_exec

  # Filter and average
  filtered_results, avg = eliminate_outliers(results)

  # write data and read data
  write_data(filtered_results, avg, model_states_sizes, methods)
  data_filtred_result, data_avg, data_state_size, data_methods = read_data()


  # Plot results
  plot_filtered_results(data_filtred_result, data_avg, data_state_size, data_methods)


def benchmark_only_plot() :
  data_filtred_result, data_avg, data_state_size, data_methods = read_data()
  plot_filtered_results(data_filtred_result, data_avg, data_state_size, data_methods)



if __name__ == "__main__":
  benchmark()
  # benchmark_only_plot()
