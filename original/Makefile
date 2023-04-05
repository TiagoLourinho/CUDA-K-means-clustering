# C compiler
HOST_CC = gcc
HOST_CC_FLAGS = -O2
HOST_CC_O3_FLAGS = -O3

GPU_CC = nvcc
GPU_CC_FLAGS = -O2 -Wno-deprecated-gpu-targets
 
DATA_DIR = ../test_data

compile_all: kmeans_host kmeans_host_O3 kmeans_gpu

compile_host: kmeans_host

compile_host_O3: kmeans_host_O3

compile_gpu: kmeans_gpu

run_host: kmeans_host
	@echo "======================================================================"
	@echo "Running with kdd_cup (without SIMD)... you may change the dataset"
	@echo "======================================================================"
	./kmeans_host -i $(DATA_DIR)/kdd_cup

run_host_O3: kmeans_host_O3
	@echo "======================================================================"
	@echo "Running with kdd_cup (with SIMD)... you may change the dataset"
	@echo "======================================================================"
	./kmeans_host_O3 -i $(DATA_DIR)/kdd_cup

run_gpu: kmeans_gpu
	@echo "======================================================================"
	@echo "Running with kdd_cup (using the GPU)... you may change the dataset"
	@echo "======================================================================"
	./kmeans_gpu -i $(DATA_DIR)/kdd_cup

run_all: kmeans_host kmeans_host_O3 kmeans_gpu
	@echo "======================================================================"
	@echo "Running with kdd_cup you may change the dataset in the makefile"
	@echo "======================================================================"
	@echo "Running on host with -O2 (standard compiler flags)"
	./kmeans_host -i $(DATA_DIR)/kdd_cup
	@echo ""
	@echo "---"
	@echo "Running on host with -O3 (autovectorization)"
	./kmeans_host_O3 -i $(DATA_DIR)/kdd_cup
	@echo ""
	@echo "---"
	@echo "Running on host (with -O2) + GPU (CUDA)"
	./kmeans_gpu -i $(DATA_DIR)/kdd_cup

########### HOST SIDE (X86) COMPILATION / no vectorization ###########
kmeans_host: cluster_host.o getopt_host.o kmeans_host.o kmeans_clustering_host.o 
	$(HOST_CC) $(HOST_CC_FLAGS) cluster_host.o getopt_host.o kmeans_host.o kmeans_clustering_host.o  -o kmeans_host

cluster_host.o: cluster.c 
	$(HOST_CC) $(HOST_CC_FLAGS) -c cluster.c -o cluster_host.o
	
getopt_host.o: getopt.c 
	$(HOST_CC) $(HOST_CC_FLAGS) -c getopt.c -o getopt_host.o
	
kmeans_host.o: kmeans.c 
	$(HOST_CC) $(HOST_CC_FLAGS) -c kmeans.c -o kmeans_host.o

kmeans_clustering_host.o: kmeans_clustering.c kmeans.h
	$(HOST_CC) $(HOST_CC_FLAGS) -c kmeans_clustering.c -o kmeans_clustering_host.o

########### HOST SIDE (X86) COMPILATION / with vectorization ###########
kmeans_host_O3: cluster_host_O3.o getopt_host_O3.o kmeans_host_O3.o kmeans_clustering_host_O3.o 
	$(HOST_CC) $(HOST_CC_O3_FLAGS) cluster_host_O3.o getopt_host_O3.o kmeans_host_O3.o kmeans_clustering_host_O3.o  -o kmeans_host_O3

cluster_host_O3.o: cluster.c 
	$(HOST_CC) $(HOST_CC_O3_FLAGS) -c cluster.c -o cluster_host_O3.o
	
getopt_host_O3.o: getopt.c 
	$(HOST_CC) $(HOST_CC_O3_FLAGS) -c getopt.c -o getopt_host_O3.o
	
kmeans_host_O3.o: kmeans.c 
	$(HOST_CC) $(HOST_CC_O3_FLAGS) -c kmeans.c -o kmeans_host_O3.o

kmeans_clustering_host_O3.o: kmeans_clustering.c kmeans.h
	$(HOST_CC) $(HOST_CC_O3_FLAGS) -c kmeans_clustering.c -o kmeans_clustering_host_O3.o

########### GPU SIDE (HOST+GPU) COMPILATION ###########
kmeans_gpu: cluster_gpu.o getopt_gpu.o kmeans_gpu.o kmeans_clustering_gpu.o 
	$(GPU_CC) $(GPU_CC_FLAGS) cluster_gpu.o getopt_gpu.o kmeans_gpu.o kmeans_clustering_gpu.o  -o kmeans_gpu

cluster_gpu.o: cluster.c 
	$(GPU_CC) $(GPU_CC_FLAGS) -c cluster.c -o cluster_gpu.o
	
getopt_gpu.o: getopt.c 
	$(GPU_CC) $(GPU_CC_FLAGS) -c getopt.c -o getopt_gpu.o
	
kmeans_gpu.o: kmeans.c 
	$(GPU_CC) $(GPU_CC_FLAGS) -c kmeans.c -o kmeans_gpu.o

kmeans_clustering_gpu.o: kmeans_clustering.c kmeans.h
	$(GPU_CC) $(GPU_CC_FLAGS) -c kmeans_clustering.c -o kmeans_clustering_gpu.o

clean:
	rm -f *.o *~ kmeans_host kmeans_host_O3 kmeans_gpu
