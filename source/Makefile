# C compiler
HOST_CC = gcc
HOST_CC_FLAGS = -O2
HOST_CC_O3_FLAGS = -O3

GPU_CC = nvcc
GPU_CC_FLAGS = -O2 -Wno-deprecated-gpu-targets 
 
DATA_DIR = ../kmeans_data
BIN_DIR = ./bin


compile_all: kmeans_host kmeans_host_O3 kmeans_gpu

compile_host: kmeans_host

compile_host_O3: kmeans_host_O3

compile_gpu: kmeans_gpu

run_host: kmeans_host
	@echo "======================================================================"
	@echo "Running with kdd_cup (without SIMD)... you may change the dataset"
	@echo "======================================================================"
	$(BIN_DIR)/kmeans_host -i $(DATA_DIR)/kdd_cup

run_host_O3: kmeans_host_O3
	@echo "======================================================================"
	@echo "Running with kdd_cup (with SIMD)... you may change the dataset"
	@echo "======================================================================"
	$(BIN_DIR)/kmeans_host_O3 -i $(DATA_DIR)/kdd_cup

run_gpu: kmeans_gpu
	@echo "======================================================================"
	@echo "Running with kdd_cup (using the GPU)... you may change the dataset"
	@echo "======================================================================"
	$(BIN_DIR)/kmeans_gpu -i $(DATA_DIR)/kdd_cup

run_all: kmeans_host kmeans_host_O3 kmeans_gpu
	@echo "======================================================================"
	@echo "Running with kdd_cup you may change the dataset in the makefile"
	@echo "======================================================================"
	@echo "Running on host with -O2 (standard compiler flags)"
	$(BIN_DIR)/kmeans_host -i $(DATA_DIR)/kdd_cup
	@echo ""
	@echo "---"
	@echo "Running on host with -O3 (autovectorization)"
	$(BIN_DIR)/kmeans_host_O3 -i $(DATA_DIR)/kdd_cup
	@echo ""
	@echo "---"
	@echo "Running on host (with -O2) + GPU (CUDA)"
	$(BIN_DIR)/kmeans_gpu -i $(DATA_DIR)/kdd_cup

########### HOST SIDE (X86) COMPILATION / no vectorization ###########
kmeans_host: cluster_host.o getopt_host.o kmeans_host.o kmeans_clustering_host.o 
	$(HOST_CC) $(HOST_CC_FLAGS) $(BIN_DIR)/cluster_host.o $(BIN_DIR)/getopt_host.o $(BIN_DIR)/kmeans_host.o $(BIN_DIR)/kmeans_clustering_host.o  -o $(BIN_DIR)/kmeans_host

cluster_host.o: cluster.c 
	$(HOST_CC) $(HOST_CC_FLAGS) -c cluster.c -o $(BIN_DIR)/cluster_host.o
	
getopt_host.o: getopt.c 
	$(HOST_CC) $(HOST_CC_FLAGS) -c getopt.c -o $(BIN_DIR)/getopt_host.o
	
kmeans_host.o: kmeans.c 
	$(HOST_CC) $(HOST_CC_FLAGS) -c kmeans.c -o $(BIN_DIR)/kmeans_host.o

kmeans_clustering_host.o: kmeans_clustering.c kmeans.h
	$(HOST_CC) $(HOST_CC_FLAGS) -c kmeans_clustering.c -o $(BIN_DIR)/kmeans_clustering_host.o

########### HOST SIDE (X86) COMPILATION / with vectorization ###########
kmeans_host_O3: cluster_host_O3.o getopt_host_O3.o kmeans_host_O3.o kmeans_clustering_host_O3.o 
	$(HOST_CC) $(HOST_CC_O3_FLAGS) $(BIN_DIR)/cluster_host_O3.o $(BIN_DIR)/getopt_host_O3.o $(BIN_DIR)/kmeans_host_O3.o $(BIN_DIR)/kmeans_clustering_host_O3.o  -o $(BIN_DIR)/kmeans_host_O3

cluster_host_O3.o: cluster.c 
	$(HOST_CC) $(HOST_CC_O3_FLAGS) -c cluster.c -o $(BIN_DIR)/cluster_host_O3.o
	
getopt_host_O3.o: getopt.c 
	$(HOST_CC) $(HOST_CC_O3_FLAGS) -c getopt.c -o $(BIN_DIR)/getopt_host_O3.o
	
kmeans_host_O3.o: kmeans.c 
	$(HOST_CC) $(HOST_CC_O3_FLAGS) -c kmeans.c -o $(BIN_DIR)/kmeans_host_O3.o

kmeans_clustering_host_O3.o: kmeans_clustering.c kmeans.h
	$(HOST_CC) $(HOST_CC_O3_FLAGS) -c kmeans_clustering.c -o $(BIN_DIR)/kmeans_clustering_host_O3.o

########### GPU SIDE (HOST+GPU) COMPILATION ###########
kmeans_gpu: cluster_gpu.o getopt_gpu.o kmeans_gpu.o kmeans_clustering_gpu.o 
	$(GPU_CC) $(GPU_CC_FLAGS) $(BIN_DIR)/cluster_gpu.o $(BIN_DIR)/getopt_gpu.o $(BIN_DIR)/kmeans_gpu.o $(BIN_DIR)/kmeans_clustering_gpu.o  -o $(BIN_DIR)/kmeans_gpu

cluster_gpu.o: cluster.c 
	$(GPU_CC) $(GPU_CC_FLAGS) -c cluster.c -o $(BIN_DIR)/cluster_gpu.o
	
getopt_gpu.o: getopt.c 
	$(GPU_CC) $(GPU_CC_FLAGS) -c getopt.c -o $(BIN_DIR)/getopt_gpu.o
	
kmeans_gpu.o: kmeans.c 
	$(GPU_CC) $(GPU_CC_FLAGS) -c kmeans.c -o $(BIN_DIR)/kmeans_gpu.o

kmeans_clustering_gpu.o: kmeans_clustering.cu kmeans.h
	$(GPU_CC) $(GPU_CC_FLAGS) -c kmeans_clustering.cu -o $(BIN_DIR)/kmeans_clustering_gpu.o

clean:
	rm -f ./bin/*
