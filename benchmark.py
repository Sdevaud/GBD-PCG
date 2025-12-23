import copy
import subprocess
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import json

def run_generator(nx, N, out_path="./include/data", nu=1, script_path="./include/generate_spd.py"):
    cmd = ["python3", script_path, str(nx), str(N), str(nu), str(out_path)]
    print(f"🛠️ Generating data: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\nstderr:\n{result.stderr}\nstdout:\n{result.stdout}")
    stdout = result.stdout.strip()
    if not stdout:
        raise RuntimeError(f"Command produced no output: {cmd}")
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip() != ""]
    try:
        x_val = float(lines[0])
    except Exception:
        raise RuntimeError(f"Invalid stdout format for command: {cmd}\nstdout:\n{stdout}\nstderr:\n{result.stderr}")
    y_val = float(lines[1]) if len(lines) >= 2 else x_val
    return x_val, y_val

def gauss_filter_xy(xs, ys):
    if len(ys) <= 2:
        return [], [], None, None
    mu = float(np.mean(ys))
    sigma = float(np.std(ys))
    if sigma == 0.0:
        return list(xs), list(ys), float(np.mean(xs)), mu
    lo = norm.ppf(0.025, loc=mu, scale=sigma)
    hi = norm.ppf(0.975, loc=mu, scale=sigma)
    fx, fy = [], []
    for x, y in zip(xs, ys):
        if lo <= y <= hi:
            fx.append(float(x))
            fy.append(float(y))
    if len(fy) == 0:
        return [], [], None, None
    return fx, fy, float(np.mean(fx)), float(np.mean(fy))

def eliminate_outliers_xy(results_x, results_y):
    nbr_run = len(results_x)
    nbr_points = len(results_x[0])
    nbr_methods = len(results_x[0][0])

    filtered_x = [[[] for _ in range(nbr_methods)] for _ in range(nbr_points)]
    filtered_y = [[[] for _ in range(nbr_methods)] for _ in range(nbr_points)]
    avg_x = [[None for _ in range(nbr_methods)] for _ in range(nbr_points)]
    avg_y = [[None for _ in range(nbr_methods)] for _ in range(nbr_points)]

    for i in range(nbr_points):
        for j in range(nbr_methods):
            xs = [results_x[k][i][j] for k in range(nbr_run)]
            ys = [results_y[k][i][j] for k in range(nbr_run)]
            fx, fy, mx, my = gauss_filter_xy(xs, ys)
            filtered_x[i][j] = fx
            filtered_y[i][j] = fy
            avg_x[i][j] = mx
            avg_y[i][j] = my

    return filtered_x, filtered_y, avg_x, avg_y

def plot_filtered_results_xy(filtered_x, filtered_y, avg_x, avg_y, method_names,
                             file_name, save_path,
                             x_label="x", y_label="y", title="Benchmark"):
    colors = plt.cm.tab10.colors
    num_methods = len(method_names)

    plt.figure(figsize=(12, 6))

    for j in range(num_methods):
        ax = [avg_x[i][j] for i in range(len(avg_x)) if avg_x[i][j] is not None and avg_y[i][j] is not None]
        ay = [avg_y[i][j] for i in range(len(avg_y)) if avg_x[i][j] is not None and avg_y[i][j] is not None]
        if len(ax) == 0:
            continue
        plt.plot(ax, ay, label=method_names[j], color=colors[j % len(colors)], marker="s", linewidth=2)
        for i in range(len(filtered_x)):
            for x, y in zip(filtered_x[i][j], filtered_y[i][j]):
                plt.scatter(x, y, color=colors[j % len(colors)], marker="x", alpha=0.6)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(title="Method")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def write_data_xy(filtered_x, filtered_y, avg_x, avg_y, methods, output_dir, file_name):
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "methods": methods,
        "filtered_x": filtered_x,
        "filtered_y": filtered_y,
        "avg_x": avg_x,
        "avg_y": avg_y,
    }
    file_path = os.path.join(output_dir, f"{file_name}.json")
    with open(file_path, "w") as f:
        json.dump(payload, f, indent=2)
    return file_path

def read_data_xy(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["filtered_x"], data["filtered_y"], data["avg_x"], data["avg_y"], data["methods"]

def _normalize_points(state_sizes, knot_points):
    if len(state_sizes) > 1 and len(knot_points) > 1 and len(state_sizes) != len(knot_points):
        raise ValueError("state_size and knot_point must match in length or one must have length 1.")
    num_points = max(len(state_sizes), len(knot_points))
    ss = list(state_sizes)
    kp = list(knot_points)
    if len(ss) == 1:
        ss = ss * num_points
    if len(kp) == 1:
        kp = kp * num_points
    return ss, kp

def _defines_from_info(info, state_size, knot_points, method_name, NBR_ITERATION_MAX=100000):
    defs = []
    defs.append(f"-DSTATE_SIZE={state_size}")
    defs.append(f"-DKNOT_POINTS={knot_points}")
    defs.append("-DDEBUG=0")

    is_optimised = "optimised" in method_name.lower()

    for name in [
        "STATExKERNEL", "KNOTxKERNEL",
        "STATExCOMPUTER", "KNOTxCOMPUTER",
        "STATExNBR_ITERATION", "KNOTxNBR_ITERATION",
        "NBR_ITERATIONxERROR", "DOUBLE",
    ]:
        val = int(getattr(info, name, 0))
        defs.append(f"-D{name}={val}")

    defs.append(f"-DOPTIMISED={int(is_optimised)}")
    defs.append(f'-DDATA_PATH=\\"{info.DATA_PATH}\\"')
    defs.append(f"-DNBR_ITERATION_MAX={NBR_ITERATION_MAX}")

    return " ".join(defs)

def _compile_method(exe_path, exe_out, defines, extra_includes_cpp="-I./include -I/usr/include/eigen3",
                    extra_includes_cu="-I./include -I./src/yang/include"):
    method_dir = os.path.dirname(exe_path)
    method_base = os.path.splitext(os.path.basename(exe_path))[0]
    cu_src = os.path.join(method_dir, f"{method_base}.cu")
    cpp_src = os.path.join(method_dir, f"{method_base}.cpp")

    subprocess.run(f"rm -f {exe_out}", shell=True)

    if os.path.exists(cu_src):
        compile_cmd = f"nvcc --compiler-options -Wall -O3 -std=c++17 {defines} {extra_includes_cu} {cu_src} -o {exe_out}"
        compiler = "nvcc"
    elif os.path.exists(cpp_src):
        compile_cmd = f"g++ -Wall -O3 -std=c++17 {defines} {extra_includes_cpp} {cpp_src} -o {exe_out}"
        compiler = "g++"
    else:
        raise FileNotFoundError(f"No .cu or .cpp file found for {exe_path}")

    print(f"🔧 Compiling ({compiler}): {compile_cmd}")
    subprocess.run(compile_cmd, shell=True, check=True)

def compute_run(info, generator_script="./include/generate_spd.py"):
    state_sizes, knot_points = _normalize_points(info.state_size, info.knot_point)
    methods = list(info.method_names)
    num_runs = int(info.nbr)

    num_points = len(state_sizes)
    num_methods = len(methods)

    results_x = [[[0.0 for _ in range(num_methods)] for _ in range(num_points)] for _ in range(num_runs)]
    results_y = [[[0.0 for _ in range(num_methods)] for _ in range(num_points)] for _ in range(num_runs)]

    prepared_cmds = [[None for _ in range(num_methods)] for _ in range(num_points)]
    executables_to_cleanup = []

    for point_idx in range(num_points):
        ss = state_sizes[point_idx]
        kp = knot_points[point_idx]
        print(f"📏 Benchmarking state_size={ss}, kp={kp}")

        for method_idx, method_name in enumerate(methods):
            src_path = info.method_path[method_name]
            defines = _defines_from_info(info, ss, kp, method_name)

            if src_path.endswith(".py"):
                prepared_cmds[point_idx][method_idx] = f"python3 {src_path} {ss} {kp} {int(getattr(info, 'STATExKERNEL', 0))} {int(getattr(info, 'KNOTxKERNEL', 0))} {int(getattr(info, 'STATExCOMPUTER', 0))} {int(getattr(info, 'KNOTxCOMPUTER', 0))} "
                continue

            method_dir = os.path.dirname(src_path)
            method_base = os.path.splitext(os.path.basename(src_path))[0]
            is_optimised = "optimised" in method_name.lower()
            opt_suffix = "opt" if is_optimised else "noopt"
            exe_out = os.path.join(
                method_dir,
                f"{method_base}_{ss}_{kp}_{opt_suffix}"
            )
            _compile_method(src_path, exe_out, defines)

            executables_to_cleanup.append(exe_out)
            prepared_cmds[point_idx][method_idx] = f"{exe_out} {info.DATA_PATH}"

        for run_idx in range(num_runs):
            print(f"🧪 Run {run_idx + 1}/{num_runs}")
            run_generator(ss, kp, out_path=info.DATA_PATH, nu=1, script_path=generator_script)

            for method_idx, method_name in enumerate(methods):
                cmd = prepared_cmds[point_idx][method_idx]
                print(f"▶️ Execution: {cmd}")
                x_val, y_val = run_cmd(cmd)
                results_x[run_idx][point_idx][method_idx] = x_val
                results_y[run_idx][point_idx][method_idx] = y_val
                print(f"⏱️  x = {x_val:.6f}, y = {y_val:.6f}\n")

    for exe in set(executables_to_cleanup):
        subprocess.run(f"rm -f {exe}", shell=True)

    return results_x, results_y

def compute_run_2(info, generator_script="./include/generate_spd.py"):
    ss, kp = info.state_size[0], info.knot_point[0]
    methods = list(info.method_names)
    num_runs = int(info.nbr)

    num_points = len(info.NBR_RUN_ITERATION)
    num_methods = len(methods)

    results_x = [[[0.0 for _ in range(num_methods)] for _ in range(num_points)] for _ in range(num_runs)]
    results_y = [[[0.0 for _ in range(num_methods)] for _ in range(num_points)] for _ in range(num_runs)]

    prepared_cmds = [[None for _ in range(num_methods)] for _ in range(num_points)]
    executables_to_cleanup = []
    run_generator(ss, kp, out_path=info.DATA_PATH, nu=1, script_path=generator_script)

    for point_idx in range(num_points):
        print(f"📏 Benchmarking nbr_iter_max : {info.NBR_RUN_ITERATION[point_idx]}")

        for method_idx, method_name in enumerate(methods):
            src_path = info.method_path[method_name]
            defines = _defines_from_info(info, ss, kp, method_name, NBR_ITERATION_MAX=info.NBR_RUN_ITERATION[point_idx])

            if src_path.endswith(".py"):
                prepared_cmds[point_idx][method_idx] = f"python3 {src_path} {ss} {kp} {int(getattr(info, 'STATExKERNEL', 0))} {int(getattr(info, 'KNOTxKERNEL', 0))} {int(getattr(info, 'STATExCOMPUTER', 0))} {int(getattr(info, 'KNOTxCOMPUTER', 0))} "
                continue

            method_dir = os.path.dirname(src_path)
            method_base = os.path.splitext(os.path.basename(src_path))[0]
            is_optimised = "optimised" in method_name.lower()
            opt_suffix = "opt" if is_optimised else "noopt"
            exe_out = os.path.join(
                method_dir,
                f"{method_base}_{ss}_{kp}_{opt_suffix}"
            )
            _compile_method(src_path, exe_out, defines)

            executables_to_cleanup.append(exe_out)
            prepared_cmds[point_idx][method_idx] = f"{exe_out} {info.DATA_PATH}"

        for run_idx in range(num_runs):
            print(f"🧪 Run {run_idx + 1}/{num_runs}")

            for method_idx, method_name in enumerate(methods):
                cmd = prepared_cmds[point_idx][method_idx]
                print(f"▶️ Execution: {cmd}")
                x_val, y_val = run_cmd(cmd)
                results_x[run_idx][point_idx][method_idx] = x_val
                results_y[run_idx][point_idx][method_idx] = y_val
                print(f"⏱️  x = {x_val:.6f}, y = {y_val:.6f}\n")

    for exe in set(executables_to_cleanup):
        subprocess.run(f"rm -f {exe}", shell=True)

    return results_x, results_y

def save_data_plot(info, generator_script="./include/generate_spd.py"):
    if info.NBR_ITERATIONxERROR == 1:
        results_x, results_y = compute_run_2(info, generator_script=generator_script)
    else:
        results_x, results_y = compute_run(info, generator_script=generator_script)
    filtered_x, filtered_y, avg_x, avg_y = eliminate_outliers_xy(results_x, results_y)

    data_dir = os.path.join(info.path_data)
    json_path = write_data_xy(filtered_x, filtered_y, avg_x, avg_y, list(info.method_names), data_dir, info.file_name_plot)

    plot_dir = os.path.join(info.path_plot)
    plot_path = os.path.join(plot_dir, f"{info.file_name_plot}.png")

    fx, fy, ax, ay, methods = read_data_xy(json_path)
    plot_filtered_results_xy(
        fx, fy, ax, ay, methods,
        file_name=f"{info.file_name_plot}.png",
        save_path=plot_path,
        x_label=info.x_label,
        y_label=info.y_label,
        title=info.title_plot
    )

class InfoBench:
  def __init__(self, 
               nbr_run,
               method_names,
               method_path,
               state_size,
               knot_point,
               path_plot,
               file_name_plot,
               title_plot,
               x_label,
               y_label,
               path_data,
               STATExKERNEL=0,
               KNOTxKERNEL=0,
               STATExCOMPUTER=0,
               KNOTxCOMPUTER=0,
               STATExNBR_ITERATION=0,
               KNOTxNBR_ITERATION=0,
               NBR_ITERATIONxERROR=0,
               OPTIMISED=0,
               DATA_PATH="./include/data",
               NBR_ITERATION_MAX=100000,
               NBR_RUN_ITERATION=[],
               DOUBLE=1):
    self.nbr = nbr_run
    self.method_names = method_names
    self.method_path = method_path
    self.state_size = state_size
    self.knot_point = knot_point
    self.path_plot = path_plot
    self.file_name_plot = file_name_plot
    self.title_plot = title_plot
    self.x_label = x_label
    self.y_label = y_label
    self.path_data = path_data
    self.STATExKERNEL = STATExKERNEL
    self.KNOTxKERNEL = KNOTxKERNEL
    self.STATExCOMPUTER = STATExCOMPUTER
    self.KNOTxCOMPUTER = KNOTxCOMPUTER
    self.STATExNBR_ITERATION = STATExNBR_ITERATION
    self.KNOTxNBR_ITERATION = KNOTxNBR_ITERATION
    self.NBR_ITERATIONxERROR = NBR_ITERATIONxERROR
    self.OPTIMISED = OPTIMISED
    self.DATA_PATH = DATA_PATH
    self.NBR_ITERATION_MAX = NBR_ITERATION_MAX
    self.NBR_RUN_ITERATION = NBR_RUN_ITERATION
    self.DOUBLE = DOUBLE

  def benchmark(self):
    save_data_plot(self)


if __name__ == "__main__":
  # ------------ First Benchmark ------------------
  benchmark1 = InfoBench(
      nbr_run = 5,
      method_names = ["numpy", "eigen", "yang_no_precond", "yang_precond", "yang_precond_optimised"],
      method_path = {
          "numpy": "./src/numpy_method.py",
          "eigen": "./src/eigen/eigen.cpp",
          "yang_no_precond": "./src/yang/yang_no_precond.cu",
          "yang_precond": "./src/yang/yang_precond.cu",
          "yang_precond_optimised": "./src/yang/yang_precond.cu"
      },
      state_size = [7 * i for i in range(1, 6)],
      knot_point = [50],
      path_plot = "plots/",
      file_name_plot = "state_kernel",
      title_plot = "Benchmark of the number of states with horizon = 50 (Kernel time)",
      x_label = "increase the number of states",
      y_label = "Execution time [ms]",
      path_data = "datas/",
      STATExKERNEL = 1
  )
  # benchmark1.benchmark()
  
  # ------------ Second Benchmark ------------------
  benchmark2 = copy.deepcopy(benchmark1)
  benchmark2.state_size = [28]
  benchmark2.knot_point = [15 * i for i in range(1, 7)]
  benchmark2.file_name_plot = "horizon_kernel"
  benchmark2.title_plot = "Benchmark of the horizon with state size = 30 (Kernel time)"
  benchmark2.x_label = "increase the horizon"
  benchmark2.STATExKERNEL = 0
  benchmark2.KNOTxKERNEL = 1
  # benchmark2.benchmark()

  # ------------ third Benchmark ------------------
  benchmark3 = copy.deepcopy(benchmark1)
  benchmark3.file_name_plot = "state_computer"
  benchmark3.title_plot = "Benchmark of the number of states with horizon = 50 (Total time)"
  benchmark3.STATExKERNEL = 0
  benchmark3.STATExCOMPUTER = 1
  # benchmark3.benchmark()

  # ------------ fourth Benchmark ------------------
  benchmark4 = copy.deepcopy(benchmark2)
  benchmark4.file_name_plot = "horizon_computer"
  benchmark4.title_plot = "Benchmark of the horizon with state size = 30 (Total time)"
  benchmark4.KNOTxKERNEL = 0
  benchmark4.KNOTxCOMPUTER = 1
  # benchmark4.benchmark()

  # ------------ fifth Benchmark ------------------
  benchmark5 = copy.deepcopy(benchmark1)
  benchmark5.method_names = ["yang_no_precond", "yang_precond"]
  benchmark5.method_path = {
          "yang_no_precond": "./src/yang/yang_no_precond.cu",
          "yang_precond": "./src/yang/yang_precond.cu"
  }
  benchmark5.file_name_plot = "state_nbr_iteration"
  benchmark5.title_plot = "Benchmark of the number of states with horizon = 50 (number of iterations)"
  benchmark5.STATExKERNEL = 0
  benchmark5.STATExNBR_ITERATION = 1
  benchmark5.y_label = "number of iterations"
  # benchmark5.benchmark()

  # ------------ sixth Benchmark ------------------
  benchmark6 = copy.deepcopy(benchmark2)
  benchmark6.file_name_plot = "horizon_nbr_iteration"
  benchmark6.method_names = ["yang_no_precond", "yang_precond"]
  benchmark6.method_path = {
          "yang_no_precond": "./src/yang/yang_no_precond.cu",
          "yang_precond": "./src/yang/yang_precond.cu"
  }
  benchmark6.title_plot = "Benchmark of the horizon with state size = 30 (number of iterations)"
  benchmark6.KNOTxKERNEL = 0
  benchmark6.KNOTxNBR_ITERATION = 1
  benchmark6.y_label = "number of iterations"
  # benchmark6.benchmark()

  # ------------ seventh Benchmark ------------------
  benchmark7 = InfoBench(
    nbr_run = 5,
    method_names = ["yang_no_precond", "yang_precond"],
    method_path = {
            "yang_no_precond": "./src/yang/yang_no_precond.cu",
            "yang_precond": "./src/yang/yang_precond.cu"
    },
    state_size = [21],
    knot_point = [80],
    path_plot = "plots/",
    file_name_plot = "errors_iterations",
    title_plot = "Benchmark of the number of iterations vs error with state size = 21 and horizon = 80",
    x_label = "number of iterations",
    y_label = "error l2 norm",
    path_data = "datas/",
    NBR_ITERATIONxERROR = 1,
    NBR_RUN_ITERATION = [20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 3500, 4000]
  )
  benchmark7.benchmark()