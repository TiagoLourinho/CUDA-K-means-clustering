import os
import subprocess
import numpy as np


GPU_DIR = os.path.join("accelerated/")
HOST_DIR = os.path.join("original/")
DATA_DIR = os.path.join("kmeans_data/")
GPU_FILE = os.path.join("bin/kmeans_gpu")
HOST_FILE = os.path.join("bin/kmeans_host_O3")
COMPILE_GPU_CMD = "make compile_gpu"
COMPILE_HOST_CMD = "make compile_host_O3"
N_RUNS = 2


def key_sort(file):
    """Used to sort test files"""
    try:
        val = int(file[: file.find("_")])

        if "f" in file:
            val += 1

        return val
    except ValueError:
        return 0


def execute_command(cmd: str, cwd: str = os.path.join(".")):
    """Executed a command in shell and returns the output"""
    return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True).stdout.decode()


def get_error(host_centers, gpu_centers):
    """Returns the error of clusters centers"""

    error = 0
    nclusters, nfeatures = host_centers.shape
    for i in range(nclusters):
        for j in range(nfeatures):
            error += abs(gpu_centers[i, j] - host_centers[i, j])

    return error


def average_runs(runs: np.array):
    """Average the values of each run"""

    n_runs = len(runs)
    nclusters = runs[0]["nclusters"]
    nfeatures = runs[0]["nfeatures"]

    time = 0
    for i in range(n_runs):
        time += runs[i]["time"]
    time = time / n_runs

    clusters = np.empty((nclusters, nfeatures), dtype=float)

    clusters_runs = np.array([runs[i]["clusters"] for i in range(n_runs)])
    for i in range(nclusters):
        for j in range(nfeatures):
            clusters[i, j] = np.average(clusters_runs[:, i, j])

    return {
        "time": time,
        "nclusters": nclusters,
        "nfeatures": nfeatures,
        "clusters": clusters,
    }


def get_run_info(out: str):
    """Given the output of the clustering, extract the relevant information"""
    lines = out.split("\n")

    nclusters = int(lines[1].split()[-1])
    nfeatures = int(lines[2].split()[-1])

    # Format of each line
    # cluster: feature1 feature2 ...
    clusters = np.empty((nclusters, nfeatures), dtype=float)
    start_line = 8
    for i in range(start_line, start_line + 2 * nclusters - 1, 2):
        values = lines[i].split()

        cluster_id = int(values[0][:-1])  # First value is the id
        for j, f in enumerate(values[1:]):
            clusters[cluster_id, j] = float(f)

    time = float(lines[-2].split()[-2])

    return {
        "time": time,
        "nclusters": nclusters,
        "nfeatures": nfeatures,
        "clusters": clusters,
    }


def main():
    # Init
    print("Compiling...")
    execute_command("make clean", cwd=GPU_DIR)
    execute_command("mkdir -p bin", cwd=GPU_DIR)
    execute_command(COMPILE_GPU_CMD, cwd=GPU_DIR)
    execute_command("make clean", cwd=HOST_DIR)
    execute_command("mkdir -p bin", cwd=HOST_DIR)
    execute_command(COMPILE_HOST_CMD, cwd=HOST_DIR)

    tests = sorted(os.listdir(DATA_DIR), key=key_sort)
    results = dict()

    with open("results.txt", "a") as file:
        for t in tests:
            print(f"Testing {t}")
            results[t] = {"host_time": None, "gpu_time": None, "error": None}

            ## Host
            runs = list()
            for _ in range(N_RUNS):
                command = (
                    f"{os.path.join(HOST_DIR,HOST_FILE)} -i {os.path.join(DATA_DIR,t)}"
                )
                output = execute_command(command)

                file.write(command + "\n")
                file.write(output + "\n")

                runs.append(get_run_info(output))

            host_average_run = average_runs(runs)

            ## GPU
            runs = list()
            for _ in range(N_RUNS):
                command = (
                    f"{os.path.join(GPU_DIR,GPU_FILE)} -i {os.path.join(DATA_DIR,t)}"
                )
                output = execute_command(command)

                file.write(command + "\n")
                file.write(output + "\n")

                runs.append(get_run_info(output))

            gpu_average_run = average_runs(runs)

            results[t]["host_time"] = host_average_run["time"]
            results[t]["gpu_time"] = gpu_average_run["time"]
            results[t]["error"] = get_error(
                host_average_run["clusters"], gpu_average_run["clusters"]
            )
            results[t]["speedup"] = host_average_run["time"] / gpu_average_run["time"]

    # Print results
    print(
        f"\n{'Test':<20}{'Host O3 Time [ms]':<20}{'GPU Time [ms]':<20}{'Total Error':<20}{'Speedup':<20}\n"
    )
    for t in tests:
        print(
            f"{t:<20}{round(results[t]['host_time'],3):<20}{round(results[t]['gpu_time'],3):<20}{round(results[t]['error'],3):<20}{round(results[t]['speedup'],3):<20}"
        )


if __name__ == "__main__":
    main()
