import subprocess

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return float(result.stdout.strip())

def gauss_filter(data):
    if len(data) <= 2:
        return None, None
    mu, sigma = np.mean(data), np.std(data)
    lower_bound = norm.ppf(0.025, loc=mu, scale=sigma)
    upper_bound = norm.ppf(0.975, loc=mu, scale=sigma)
    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
    return filtered_data, np.mean(filtered_data)

def eliminate_outliers(results):
    filtered_results = [[[]for _ in range(len(iteration))]for _ in range(len(nbr_run))]
    avg = [[]for _ in range(N)]
    for i in range(len(results[0])):
        for j in range(len(results[0][0])):
          all_run = []
          all_run.extend(results[k][i][j] for k in range(len(results)))
          filtered_data, mean = gauss_filter(all_run)
          if filtered_data is not None:
            filtered_results[i][j] = filtered_data
    return filtered_results


def benchmark(nbr_run):

  N = 1
  n = 15
  iteration = 6
  ## declaration of the result list
  results = [[[]for _ in range(len(iteration))]for _ in range(len(nbr_run))]
  ## run example
  for i in range(nbr_run):
    for j in range(iteration):
      gauss_jordan_time = subprocess.run(["path", "N * iteration * 10 + 1"])
      results[i][j].extend(gauss_jordan_time)

      numpy_time = subprocess.run(["path", "N * iteration * 10 + 1"])
      results[i][j].extend(numpy_time)

      gradient_time_no_gpu = subprocess.run(["path", "N * iteration * 10 + 1"])
      results[i][j].extend(gradient_time_no_gpu)

      gradient_time_gpu_no_precond = subprocess.run(["path", "N * iteration * 10 + 1"])
      results[i][j].extend(gradient_time_gpu_no_precond)

      gradient_time_gpu_precond = subprocess.run(["path", "N * iteration * 10 + 1"])
      results[i][j].extend(gradient_time_gpu_precond)


  






if __name__ == "__main__":
    benchmark(nbr_run, N, n, dense, Type, precond)