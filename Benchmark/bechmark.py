import subprocess
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os


def compile_all(method_paths):
    """
    Compile automatiquement tous les dossiers CUDA présents dans method_paths.

    Parameters
    ----------
    method_paths : dict
        Dictionnaire {nom_methode: chemin}
        où chemin peut être un script Python (.py) ou un exécutable CUDA (.exe)
    """
    compiled = []

    for method, path in method_paths.items():
        # Si c’est un script Python, on ne compile pas
        if path.endswith(".py"):
            print(f"🟢 Méthode '{method}' (Python) — pas de compilation nécessaire.")
            continue

        # Si c’est un exécutable CUDA dans un dossier (ex: ./CG_no_GPU/CG_no_GPU.exe)
        dir_path = os.path.dirname(path)
        if not dir_path:
            print(f"⚠️ Impossible de déterminer le dossier pour {path}")
            continue

        print(f"🔧 Compilation du code CUDA pour '{method}' dans {dir_path} ...")

        try:
            subprocess.run(["make", "-C", dir_path], check=True)
            compiled.append(dir_path)
        except subprocess.CalledProcessError:
            print(f"❌ Erreur lors de la compilation de {dir_path}")

    if compiled:
        print(f"✅ Compilation terminée pour : {', '.join(compiled)}\n")
    else:
        print("🟢 Aucune compilation CUDA nécessaire.\n")

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

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"✅ Figure sauvegardée dans : {save_path}")
    else:
        plt.show()

    plt.close()

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

  # Plot results
  plot_filtered_results(filtered_results, avg, model_states_sizes, methods, save_path="/home/sdevaud/Semester_Project/GBD-PCG/Benchmark/plots/bench.png")




if __name__ == "__main__":
  benchmark()