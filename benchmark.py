from include.utils_benchmark import InfoBench
from include.utils_benchmark import create_benchmark
  

if __name__ == "__main__":
  nbr_run_for_variance = 50
  benchmarks = create_benchmark(nbr_run_for_variance)

  # benchmarks[0].benchmark()
  # benchmarks[1].benchmark()
  # benchmarks[2].benchmark()
  # benchmarks[3].benchmark()
  # benchmarks[4].benchmark()
  # benchmarks[5].benchmark()
  # benchmarks[6].benchmark()
  # benchmarks[7].benchmark()
  # benchmarks[8].benchmark()
  # benchmarks[9].benchmark()
  # benchmarks[10].benchmark()
  # benchmarks[11].benchmark()

  benchmarks[0].only_plot()
  benchmarks[1].only_plot()
  benchmarks[2].only_plot()
  benchmarks[3].only_plot()
  benchmarks[4].only_plot()
  benchmarks[5].only_plot()
  benchmarks[6].only_plot()
  benchmarks[7].only_plot()
  benchmarks[8].only_plot()
  benchmarks[9].only_plot()
  benchmarks[10].only_plot()
  benchmarks[11].only_plot()
