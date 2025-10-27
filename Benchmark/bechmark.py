import subprocess
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import json


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

def write_data(filtered_results, avg, model_states_sizes, methods, output_dir="data"):
    """
    Sauvegarde les données filtrées et moyennes dans un dossier (un fichier par méthode).
    Les fichiers sont écrits en JSON pour plus de portabilité.
    """
    os.makedirs(output_dir, exist_ok=True)

    for method_idx, method in enumerate(methods):
        method_data = {
            "model_states_sizes": model_states_sizes,
            "filtered_results": [  # extrait uniquement les temps pour cette méthode
                [run[i][method_idx] for i in range(len(model_states_sizes))]
                for run in filtered_results
            ],
            "avg": [avg[i][method_idx] for i in range(len(model_states_sizes))]
        }

        file_path = os.path.join(output_dir, f"{method}.json")
        with open(file_path, "w") as f:
            json.dump(method_data, f, indent=2)

        print(f"💾 Données sauvegardées dans {file_path}")

def read_data(data_dir="data"):
    """
    Relit les données depuis le dossier `data/` et reconstruit les structures :
      - filtered_results
      - avg
      - model_states_sizes
      - methods
    pour pouvoir replotter sans relancer la simulation.
    """
    files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    if not files:
        raise FileNotFoundError(f"Aucun fichier de données trouvé dans {data_dir}")

    methods = []
    model_states_sizes = None
    filtered_results_per_method = {}
    avg_per_method = {}

    for filename in files:
        method = os.path.splitext(filename)[0]
        methods.append(method)

        with open(os.path.join(data_dir, filename), "r") as f:
            data = json.load(f)

        if model_states_sizes is None:
            model_states_sizes = data["model_states_sizes"]

        filtered_results_per_method[method] = data["filtered_results"]
        avg_per_method[method] = data["avg"]

    # Reconstruction dans le même format que l’original :
    nbr_run = len(next(iter(filtered_results_per_method.values())))
    nbr_sizes = len(model_states_sizes)
    nbr_methods = len(methods)

    # filtered_results[run][i][j]
    filtered_results = [
        [
            [filtered_results_per_method[methods[j]][run][i] for j in range(nbr_methods)]
            for i in range(nbr_sizes)
        ]
        for run in range(nbr_run)
    ]

    avg = [
        [avg_per_method[methods[j]][i] for j in range(nbr_methods)]
        for i in range(nbr_sizes)
    ]

    print(f"📂 Données chargées depuis {data_dir} ({len(methods)} méthodes, {nbr_run} runs)")
    return filtered_results, avg, model_states_sizes, methods

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
  write_data(filtered_results, avg, model_states_sizes, methods)
  data_filtred_result, data_avg, data_state_size, data_methods = read_data()


  # Plot results
  plot_filtered_results(data_filtred_result, data_avg, data_state_size, data_methods, save_path="/home/sdevaud/Semester_Project/GBD-PCG/Benchmark/plots/bench.png")




if __name__ == "__main__":
  benchmark()