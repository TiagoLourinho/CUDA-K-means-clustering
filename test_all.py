""" 
Python script that makes N_RUNS per test and device, averaging the time results
Presents a results table with times, error and obtained speedup

The config parameters below should be adjusted accordingly:
|
|
V
"""


import os
import subprocess

# Config parameters
SOURCE_DIR = os.path.join("source/")
DATA_DIR = os.path.join("kmeans_data/")
GPU_FILE = os.path.join(SOURCE_DIR, "bin/kmeans_gpu")
HOST_FILE = os.path.join(SOURCE_DIR, "bin/kmeans_host_O3")
COMPILE_GPU_CMD = "make compile_gpu"
COMPILE_HOST_CMD = "make compile_host_O3"
N_RUNS = 5


def key_sort(file: str) -> int:
    """Used to sort test files"""
    try:
        val = int(file[: file.find("_")])

        if "f" in file:
            val += 1

        return val
    except ValueError:
        return 0


def execute_command(cmd: str, cwd: str = os.path.join(".")) -> str:
    """Executed a command in shell and returns the output"""
    return subprocess.run(
        cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE
    ).stdout.decode()


def get_error(host_centers: list, gpu_centers: list) -> float:
    """Returns the error of clusters centers"""

    error = 0
    nclusters = len(host_centers)
    nfeatures = len(host_centers[0])
    for i in range(nclusters):
        for j in range(nfeatures):
            error += abs(gpu_centers[i][j] - host_centers[i][j])

    return error


def average_runs(runs: list) -> dict:
    """Average the values of each run"""

    n_runs = len(runs)
    nclusters = runs[0]["nclusters"]
    nfeatures = runs[0]["nfeatures"]

    time = 0
    for i in range(n_runs):
        time += runs[i]["time"]
    time = time / n_runs

    clusters_runs = [runs[i]["clusters"] for i in range(n_runs)]

    return {
        "time": time,
        "nclusters": nclusters,
        "nfeatures": nfeatures,
        "clusters": average_along_first_dim(clusters_runs),
    }


def average_along_first_dim(data: list) -> list:
    """Calculates the average along the first axis of a 3D list"""

    height, width, depth = len(data), len(data[0]), len(data[0][0])
    result = []
    for i in range(width):
        row = []
        for j in range(depth):
            total = 0
            for k in range(height):
                total += data[k][i][j]
            row.append(total / height)
        result.append(row)
    return result


def get_run_info(out: str) -> dict:
    """Given the output of the clustering, extract the relevant information"""
    lines = out.split("\n")

    nclusters = int(lines[1].split()[-1])
    nfeatures = int(lines[2].split()[-1])

    # Format of each line
    # cluster: feature1 feature2 ...
    clusters = list()
    start_line = 8
    for i in range(start_line, start_line + 2 * nclusters - 1, 2):
        values = lines[i].split()

        clusters.append(list())

        cluster_id = int(values[0][:-1])  # First value is the id
        for feature in values[1:]:
            clusters[cluster_id].append(float(feature))

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
    execute_command("make clean", cwd=SOURCE_DIR)
    execute_command("mkdir -p bin", cwd=SOURCE_DIR)
    execute_command(COMPILE_GPU_CMD, cwd=SOURCE_DIR)
    execute_command(COMPILE_HOST_CMD, cwd=SOURCE_DIR)

    tests = sorted(os.listdir(DATA_DIR), key=key_sort)
    results = dict()

    with open("results.txt", "w") as file:
        for t in tests:
            print(f"Testing {t}")
            results[t] = {"host_time": None, "gpu_time": None, "error": None}

            for turn in [HOST_FILE, GPU_FILE]:
                runs = list()
                for _ in range(N_RUNS):
                    command = f"{turn} -i {os.path.join(DATA_DIR,t)}"
                    output = execute_command(command)

                    file.write(command + "\n")
                    file.write(output + "\n")

                    if "error" in output:
                        print(
                            f"Error occurred while running test {t} in {turn}: {output[output.index('error:')+len('error:')+1:]}"
                        )

                    runs.append(get_run_info(output))

                if turn == HOST_FILE:
                    host_average_run = average_runs(runs)
                else:
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
