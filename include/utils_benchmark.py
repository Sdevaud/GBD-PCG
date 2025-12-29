import copy
import subprocess
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import json

def plot_SharedMemory():
  x = [2*i for i in range(1, 20)]

  y_64_no = [3 * i * i / 128 for i in x]
  y_64_pre = [3 * 3 * i * i / 128 for i in x]
  y_32_no = [3 * i * i / 256 for i in x]
  y_32_pre = [3 * 3 * i * i / 256 for i in x]

  plt.figure(figsize=(6, 4))

  plt.plot(x, y_64_no, marker="x", label="64-bit no precond")
  plt.plot(x, y_64_pre, marker="x", label="64-bit precond")
  plt.plot(x, y_32_no, marker="o", label="32-bit no precond")
  plt.plot(x, y_32_pre, marker="o", label="32-bit precond")

  plt.axhline(96, color="red", linestyle="--", linewidth=2, label="Shared memory limit (96 KB)")

  plt.xlabel("state size")
  plt.ylabel("KB")
  plt.title("Shared memory size vs number of state")
  plt.legend()
  plt.grid(True, linestyle="--", alpha=0.5)
  plt.tight_layout()
  os.makedirs(os.path.dirname("./plots/Shared_memroy"), exist_ok=True)
  plt.savefig("./plots/Shared_memroy", dpi=300, bbox_inches="tight")
  plt.close()

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
    lo = norm.ppf(0.1, loc=mu, scale=sigma)
    hi = norm.ppf(0.9, loc=mu, scale=sigma)
    fx, fy = [], []
    for x, y in zip(xs, ys):
        if lo <= y <= hi:
            fx.append(float(x))
            fy.append(float(y))
    if len(fy) == 0:
        return [], [], None, None
    return fx, fy, float(np.mean(fx)), float(np.mean(fy))

def eliminate_outliers_xy(results_x, results_y, enable_stop_on_drop=False):
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

    if enable_stop_on_drop:
      for j in range(nbr_methods):
        prev = None
        stop = False
        for i in range(nbr_points):
          curr = avg_y[i][j]

          if stop or curr is None or prev is None:
            if stop:
              avg_x[i][j] = None
              avg_y[i][j] = None
              filtered_x[i][j] = []
              filtered_y[i][j] = []
            prev = curr if curr is not None else prev
            continue

          if curr < (1.0 - 0.3) * prev:
            stop = True
            avg_x[i][j] = None
            avg_y[i][j] = None
            filtered_x[i][j] = []
            filtered_y[i][j] = []
          else:
            prev = curr

    return filtered_x, filtered_y, avg_x, avg_y

def plot_filtered_results_xy(
    filtered_x,
    filtered_y,
    avg_x,
    avg_y,
    method_names,
    file_name,
    save_path,
    plot_only_means=False,
    x_label="x",
    y_label="y",
    title="Benchmark",
    ay_right=False
):
    list_color_name = ["numpy", "eigen", "yang_no_precond", "yang_precond", "yang_precond_optimised", "gato"]

    palette = list(plt.cm.tab10.colors)

    color_map = {}
    used_colors = 0

    for name in list_color_name:
        if used_colors < len(palette):
            color_map[name] = palette[used_colors]
            used_colors += 1

    for name in method_names:
        if name not in color_map:
            color_map[name] = palette[used_colors % len(palette)]
            used_colors += 1

    plt.figure(figsize=(6, 8))

    for j, method in enumerate(method_names):
        xs = []
        ys = []
        for i in range(len(avg_x)):
            if avg_x[i][j] is not None and avg_y[i][j] is not None:
                xs.append(avg_x[i][j])
                ys.append(avg_y[i][j])

        if len(xs) == 0:
            continue

        c = color_map[method]

        plt.plot(
            xs,
            ys,
            label=method,
            color=c,
            marker="s",
            linewidth=2
        )

        if not plot_only_means:
            for i in range(len(filtered_x)):
                for x, y in zip(filtered_x[i][j], filtered_y[i][j]):
                    plt.scatter(
                        x,
                        y,
                        color=c,
                        marker="x",
                        alpha=0.6
                    )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    ax = plt.gca()

    if ay_right:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

    plt.legend(title="Method", fontsize=8)
    plt.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
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
        "NBR_ITERATIONxERROR", "KERNELxERROR", "DOUBLE",
    ]:
        val = int(getattr(info, name, 0))
        defs.append(f"-D{name}={val}")

    defs.append(f"-DOPTIMISED={int(is_optimised)}")
    defs.append(f'-DDATA_PATH=\\"{info.DATA_PATH}\\"')
    defs.append(f"-DNBR_ITERATION_MAX={NBR_ITERATION_MAX}")

    return " ".join(defs)

def _compile_method(exe_path, exe_out, defines, extra_includes_cpp="-I./include -I/usr/include/eigen3",
                    extra_includes_cu="-I./include -I./src/yang/include -I./src/gato/include"):
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
    num_runs = int(info.nbr_run)

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
                prepared_cmds[point_idx][method_idx] = f"python3 {src_path} {ss} {kp} {int(getattr(info, 'STATExKERNEL', 0))} {int(getattr(info, 'KNOTxKERNEL', 0))} {int(getattr(info, 'STATExCOMPUTER', 0))} {int(getattr(info, 'KNOTxCOMPUTER', 0))} {int(getattr(info, 'KERNELxERROR', 0))}"
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
    num_runs = int(info.nbr_run)

    num_points = len(info.NBR_RUN_ITERATION)
    num_methods = len(methods)

    results_x = [[[0.0 for _ in range(num_methods)] for _ in range(num_points)] for _ in range(num_runs)]
    results_y = [[[0.0 for _ in range(num_methods)] for _ in range(num_points)] for _ in range(num_runs)]

    prepared_cmds = [[None for _ in range(num_methods)] for _ in range(num_points)]
    executables_to_cleanup = []

    for run_idx in range(num_runs):
      run_generator(ss, kp, out_path=info.DATA_PATH, nu=1, script_path=generator_script)
      for point_idx in range(num_points):
        print(f"📏 Benchmarking nbr_iter_max : {info.NBR_RUN_ITERATION[point_idx]}")

        for method_idx, method_name in enumerate(methods):
          src_path = info.method_path[method_name]
          defines = _defines_from_info(info, ss, kp, method_name, NBR_ITERATION_MAX=info.NBR_RUN_ITERATION[point_idx])

          if src_path.endswith(".py"):
            prepared_cmds[point_idx][method_idx] = f"python3 {src_path} {ss} {kp} {int(getattr(info, 'STATExKERNEL', 0))} {int(getattr(info, 'KNOTxKERNEL', 0))} {int(getattr(info, 'STATExCOMPUTER', 0))} {int(getattr(info, 'KNOTxCOMPUTER', 0))} {int(getattr(info, 'KERNELxERROR', 0))} "
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
    if (info.NBR_ITERATIONxERROR == 1) or (info.KERNELxERROR == 1):
      results_x, results_y = compute_run_2(info, generator_script=generator_script)
      plot_only_means = True
    else:
      results_x, results_y = compute_run(info, generator_script=generator_script)
      plot_only_means = False
    filtered_x, filtered_y, avg_x, avg_y = eliminate_outliers_xy(results_x, results_y, info.enable_stop_on_drop)

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
        title=info.title_plot,
        plot_only_means=plot_only_means,
        ay_right=info.ay_right
    )

def only_plot(info):
  if (info.NBR_ITERATIONxERROR == 1) or (info.KERNELxERROR == 1):
    plot_only_means = True
  else:
    plot_only_means = False
  
  json_path = os.path.join(os.path.join(info.path_data), f"{info.file_name_plot}.json")
  fx, fy, ax, ay, methods = read_data_xy(json_path)
  plot_dir = os.path.join(info.path_plot)
  plot_path = os.path.join(plot_dir, f"{info.file_name_plot}.png")
  plot_filtered_results_xy(
      fx, fy, ax, ay, methods,
      file_name=f"{info.file_name_plot}.png",
      save_path=plot_path,
      x_label=info.x_label,
      y_label=info.y_label,
      title=info.title_plot,
      plot_only_means=plot_only_means,
      ay_right=info.ay_right
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
               ay_right=False,
               enable_stop_on_drop=False,
               STATExKERNEL=0,
               KNOTxKERNEL=0,
               STATExCOMPUTER=0,
               KNOTxCOMPUTER=0,
               STATExNBR_ITERATION=0,
               KNOTxNBR_ITERATION=0,
               NBR_ITERATIONxERROR=0,
               KERNELxERROR=0,
               OPTIMISED=0,
               DATA_PATH="./include/data",
               NBR_ITERATION_MAX=100000,
               NBR_RUN_ITERATION=[],
               DOUBLE=1):
    self.nbr_run = nbr_run
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
    self.ay_right = ay_right
    self.enable_stop_on_drop = enable_stop_on_drop
    self.STATExKERNEL = STATExKERNEL
    self.KNOTxKERNEL = KNOTxKERNEL
    self.STATExCOMPUTER = STATExCOMPUTER
    self.KNOTxCOMPUTER = KNOTxCOMPUTER
    self.STATExNBR_ITERATION = STATExNBR_ITERATION
    self.KNOTxNBR_ITERATION = KNOTxNBR_ITERATION
    self.NBR_ITERATIONxERROR = NBR_ITERATIONxERROR
    self.KERNELxERROR = KERNELxERROR
    self.OPTIMISED = OPTIMISED
    self.DATA_PATH = DATA_PATH
    self.NBR_ITERATION_MAX = NBR_ITERATION_MAX
    self.NBR_RUN_ITERATION = NBR_RUN_ITERATION
    self.DOUBLE = DOUBLE

  def benchmark(self):
    save_data_plot(self)

  def only_plot(self):
    only_plot(self)


def create_benchmark(nbr_run_for_variance):
  list_of_benchmarks = []
  
  # ------------ First Benchmark ------------------
  benchmark1 = InfoBench(
      nbr_run = nbr_run_for_variance,
      method_names = ["numpy", "eigen", "yang_no_precond", "yang_precond", "yang_precond_optimised", "gato"],
      method_path = {
          "numpy": "./src/numpy_method.py",
          "eigen": "./src/eigen/eigen.cpp",
          "yang_no_precond": "./src/yang/yang_no_precond.cu",
          "yang_precond": "./src/yang/yang_precond.cu",
          "yang_precond_optimised": "./src/yang/yang_precond.cu",
          "gato":"./src/gato/gato.cu"
      },
      state_size = [7 * i for i in range(1, 6)],
      knot_point = [40],
      path_plot = "plots/",
      file_name_plot = "state_kernel",
      title_plot = "States size vs Kernel Execution Time, N = 40",
      x_label = "increase the number of states",
      y_label = "Execution time [ms]",
      path_data = "datas/",
      STATExKERNEL = 1,
      enable_stop_on_drop=True
  )
  list_of_benchmarks.append(benchmark1)
  
  # ------------ Second Benchmark ------------------
  benchmark2 = copy.deepcopy(benchmark1)
  benchmark2.state_size = [21]
  benchmark2.knot_point = [15 * i for i in range(1, 7)]
  benchmark2.file_name_plot = "horizon_kernel"
  benchmark2.title_plot = "Knot Point vs Kernel Execution Time, nx = 21"
  benchmark2.x_label = "increase the number of Knot Points (horizon)"
  benchmark2.STATExKERNEL = 0
  benchmark2.KNOTxKERNEL = 1
  list_of_benchmarks.append(benchmark2)

  # ------------ third Benchmark ------------------
  benchmark3 = copy.deepcopy(benchmark1)
  benchmark3.file_name_plot = "state_computer"
  benchmark3.title_plot = "States size vs Total Execution Time, N = 40"
  benchmark3.STATExKERNEL = 0
  benchmark3.STATExCOMPUTER = 1
  list_of_benchmarks.append(benchmark3)

  # ------------ fourth Benchmark ------------------
  benchmark4 = copy.deepcopy(benchmark2)
  benchmark4.file_name_plot = "horizon_computer"
  benchmark4.title_plot = "Knot Point vs Total Execution Time, nx = 21"
  benchmark4.KNOTxKERNEL = 0
  benchmark4.KNOTxCOMPUTER = 1
  benchmark4.ay_right = True
  list_of_benchmarks.append(benchmark4)

  # ------------ fifth Benchmark ------------------
  benchmark5 = copy.deepcopy(benchmark1)
  benchmark5.method_names = ["yang_no_precond", "yang_precond", "gato"]
  benchmark5.method_path = {
          "yang_no_precond": "./src/yang/yang_no_precond.cu",
          "yang_precond": "./src/yang/yang_precond.cu",
          "gato":"./src/gato/gato.cu"
  }
  benchmark5.file_name_plot = "state_nbr_iteration"
  benchmark5.title_plot = "State vs Number of Iterations, N = 40"
  benchmark5.ay_right = True
  benchmark5.STATExKERNEL = 0
  benchmark5.STATExNBR_ITERATION = 1
  benchmark5.y_label = "number of iterations"
  list_of_benchmarks.append(benchmark5)

  # ------------ sixth Benchmark ------------------
  benchmark6 = copy.deepcopy(benchmark2)
  benchmark6.file_name_plot = "horizon_nbr_iteration"
  benchmark6.method_names = benchmark5.method_names
  benchmark6.method_path = benchmark5.method_path
  benchmark6.title_plot = "Knot Point vs Number of Iterations, nx = 21"
  benchmark6.ay_right = True
  benchmark6.KNOTxKERNEL = 0
  benchmark6.KNOTxNBR_ITERATION = 1
  benchmark6.y_label = "number of iterations"
  list_of_benchmarks.append(benchmark6)

  # ------------ seventh Benchmark ------------------
  benchmark7 = InfoBench(
    nbr_run = benchmark1.nbr_run,
    method_names = benchmark5.method_names,
    method_path = benchmark5.method_path,
    state_size = [21],
    knot_point = [40],
    path_plot = "plots/",
    file_name_plot = "errors_iterations_21_40",
    title_plot = "Number of iterations vs error : nx = 21,  N = 40,  max_iter = 840",
    x_label = "number of iterations",
    y_label = "error L2 norm",
    path_data = "datas/",
    ay_right = True,
    NBR_ITERATIONxERROR = 1,
    NBR_RUN_ITERATION = [0, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
  )
  list_of_benchmarks.append(benchmark7)

   # ------------ eighth Benchmark ------------------
  benchmark8 = copy.deepcopy(benchmark7)
  benchmark8.state_size = [30]
  benchmark8.knot_point = [90]
  benchmark8.file_name_plot = "errors_iterations_30_90"
  benchmark8.title_plot = "Number of iterations vs error : nx = 30, N = 90 max_iter = 2700"
  benchmark8.ay_right = True
  benchmark8.NBR_RUN_ITERATION = [0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000]
  list_of_benchmarks.append(benchmark8)

  # ------------ ninth Benchmark ------------------
  benchmark9 = copy.deepcopy(benchmark7)
  benchmark9.method_names = ["numpy", "eigen", "yang_no_precond", "yang_precond"]
  benchmark9.method_path = {
      "numpy": "./src/numpy_method.py",
      "eigen": "./src/eigen/eigen.cpp",
      "yang_no_precond": "./src/yang/yang_no_precond.cu",
      "yang_precond": "./src/yang/yang_precond.cu"
  }
  benchmark9.file_name_plot = "errors_kernel_21_40"
  benchmark9.title_plot = "Kernel time vs error : nx = 21,  N = 40,  max_iter = 840"
  benchmark9.ay_right = False
  benchmark9.NBR_ITERATIONxERROR = 0
  benchmark9.KERNELxERROR = 1
  benchmark9.x_label = "Kernel Execution Time [ms]"
  list_of_benchmarks.append(benchmark9)

  # ------------ tenth Benchmark ------------------
  benchmark10 = copy.deepcopy(benchmark8)
  benchmark10.method_names = benchmark9.method_names
  benchmark10.method_path = benchmark9.method_path
  benchmark10.file_name_plot = "errors_kernel_30_90"
  benchmark10.title_plot = "Kernel time vs error : nx = 30, N = 90 max_iter = 2700"
  benchmark10.ay_right = False
  benchmark10.NBR_ITERATIONxERROR = 0
  benchmark10.KERNELxERROR = 1
  benchmark10.x_label = benchmark9.x_label
  list_of_benchmarks.append(benchmark10)

  # ------------ eleventh Benchmark ------------------
  benchmark11 = InfoBench(
    nbr_run = benchmark1.nbr_run,
    method_names = benchmark5.method_names,
    method_path = benchmark5.method_path,
    state_size = [9],
    knot_point = [16],
    path_plot = "plots/",
    file_name_plot = "errors_iterations_9_16",
    title_plot = "Number of iterations vs error : nx = 9, N = 16 max_iter = 144",
    x_label = "number of iterations",
    y_label = "error L2 norm",
    path_data = "datas/",
    ay_right = True,
    NBR_ITERATIONxERROR = 1,
    NBR_RUN_ITERATION = [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 144]
  )
  list_of_benchmarks.append(benchmark11)

  # ------------ twelveth Benchmark ------------------
  benchmark12 = copy.deepcopy(benchmark11)
  benchmark12.method_names = benchmark9.method_names
  benchmark12.method_path = benchmark9.method_path
  benchmark12.file_name_plot = "errors_kernel_9_16"
  benchmark12.title_plot = "Kernel time vs error : nx = 9, N = 16 max_iter = 144"
  benchmark12.ay_right = False
  benchmark12.NBR_ITERATIONxERROR = 0
  benchmark12.KERNELxERROR = 1
  benchmark12.x_label = benchmark9.x_label
  list_of_benchmarks.append(benchmark12)

  return list_of_benchmarks