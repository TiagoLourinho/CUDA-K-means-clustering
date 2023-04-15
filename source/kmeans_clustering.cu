/*****************************************************************************/
/*IMPORTANT:  READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.         */
/*By downloading, copying, installing or using the software you agree        */
/*to this license.  If you do not agree to this license, do not download,    */
/*install, copy or use the software.                                         */
/*                                                                           */
/*                                                                           */
/*Copyright (c) 2005 Northwestern University                                 */
/*All rights reserved.                                                       */

/*Redistribution of the software in source and binary forms,                 */
/*with or without modification, is permitted provided that the               */
/*following conditions are met:                                              */
/*                                                                           */
/*1       Redistributions of source code must retain the above copyright     */
/*        notice, this list of conditions and the following disclaimer.      */
/*                                                                           */
/*2       Redistributions in binary form must reproduce the above copyright   */
/*        notice, this list of conditions and the following disclaimer in the */
/*        documentation and/or other materials provided with the distribution.*/
/*                                                                            */
/*3       Neither the name of Northwestern University nor the names of its    */
/*        contributors may be used to endorse or promote products derived     */
/*        from this software without specific prior written permission.       */
/*                                                                            */
/*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS    */
/*IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED      */
/*TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT AND         */
/*FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          */
/*NORTHWESTERN UNIVERSITY OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,       */
/*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES          */
/*(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR          */
/*SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          */
/*HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,         */
/*STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN    */
/*ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             */
/*POSSIBILITY OF SUCH DAMAGE.                                                 */
/******************************************************************************/
/*************************************************************************/
/**   File:         kmeans_clustering.c                                 **/
/**   Description:  Implementation of regular k-means clustering        **/
/**                 algorithm                                           **/
/**   Author:  Wei-keng Liao                                            **/
/**            ECE Department, Northwestern University                  **/
/**            email: wkliao@ece.northwestern.edu                       **/
/**                                                                     **/
/**   Edited by: Jay Pisharath                                          **/
/**              Northwestern University.                               **/
/**                                                                     **/
/**   ================================================================  **/
/**																		**/
/**   Edited by: Sang-Ha  Lee											**/
/**				 University of Virginia									**/
/**																		**/
/**   Description:	No longer supports fuzzy c-means clustering;	 	**/
/**					only regular k-means clustering.					**/
/**					Simplified for main functionality: regular k-means	**/
/**					clustering.											**/
/**                                                                     **/
/*************************************************************************/
// To solve linking problems
extern "C"
{
#include "kmeans.h"
}

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "kmeans.h"
#include <omp.h>

#define RANDOM_MAX 2147483647

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

// CUDA Config
#define THREADS_PER_BLOCK 32
#define WARP_SIZE 32
#define TOTAL_THREAD_LIMIT 1024

#define THREAD_LIMIT_X 1024
#define THREAD_LIMIT_Y 1024
#define THREAD_LIMIT_Z 64

#define BLOCK_LIMIT_X 2147483647
#define BLOCK_LIMIT_Y 65535
#define BLOCK_LIMIT_Z 65535

#define checkError(ans)                       \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "Device error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

extern double wtime(void);

__constant__ int d_nfeatures;
__constant__ int d_npoints;
__constant__ int d_nclusters;
__constant__ float d_threshold;

/* ==================== Host util functions ==================== */

int updiv(int threads_per_block, int N)
{
    return (N + threads_per_block - 1) / threads_per_block;
}

int config_check(int npoints, int nfeatures, int nclusters)
{
    if (nfeatures > THREAD_LIMIT_X)
        return 1;

    if (nclusters > THREAD_LIMIT_Y)
        return 1;

    if (nfeatures * nclusters > TOTAL_THREAD_LIMIT)
        return 1;

    int threads_per_block_x = TOTAL_THREAD_LIMIT / nfeatures;

    if (nfeatures > THREAD_LIMIT_Y)
        return 1;

    if (threads_per_block_x > THREAD_LIMIT_X)
        return 1;

    if (nfeatures * threads_per_block_x > TOTAL_THREAD_LIMIT)
        return 1;

    if (updiv(threads_per_block_x, npoints) > BLOCK_LIMIT_X)
        return 1;

    if (updiv(THREADS_PER_BLOCK, npoints) > BLOCK_LIMIT_X)
        return 1;

    if (THREADS_PER_BLOCK > TOTAL_THREAD_LIMIT)
        return 1;

    return 0;
}

/* ==================== Init functions ==================== */

__global__ void init_cluster_centers(float *d_clusters, float *d_feature)
{
    int cluster = blockIdx.y * blockDim.y + threadIdx.y;
    int feature = blockIdx.x * blockDim.x + threadIdx.x;

    if (cluster < d_nclusters && feature < d_nfeatures)
    {
        d_clusters[d_nfeatures * cluster + feature] = d_feature[d_nfeatures * cluster + feature];
    }
}

__global__ void init_membership(int *d_membership)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < d_npoints)
    {
        d_membership[tid] = -1;
    }
}

__global__ void reset_aux_data(int *d_new_centers_len, float *d_new_centers, float *d_delta)
{
    int cluster = blockIdx.y * blockDim.y + threadIdx.y;
    int feature = blockIdx.x * blockDim.x + threadIdx.x;

    if (cluster < d_nclusters && feature < d_nfeatures)
    {
        // Only 1 thread will reset delta
        if (cluster == 0 && feature == 0)
            *d_delta = 0;

        // Only 1 thread per cluster will reset len
        if (feature == 0)
            d_new_centers_len[cluster] = 0;

        d_new_centers[cluster * d_nfeatures + feature] = 0.0;
    }
}

/* ==================== Main computational functions ==================== */

__global__ void assign_membership(float *d_feature, float *d_clusters, int *d_membership, float *d_new_centers, int *d_new_centers_len, float *d_delta, int optimized)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    int index, j, i;
    float dist, min_dist;
    float aux1, aux2;

    extern __shared__ float s_clusters[];
    __shared__ float s_delta;

    // Load to share memory
    if (optimized) // Number of threads is enough to load all data (bigger than d_nclusters * d_nfeatures)
    {
        if (local_tid < d_nclusters * d_nfeatures)
            s_clusters[local_tid] = d_clusters[local_tid];
    }
    else
    {
        if (local_tid < d_nclusters)
        {
            for (j = 0; j < d_nfeatures; j++)
            {
                s_clusters[threadIdx.x * d_nfeatures + j] = d_clusters[threadIdx.x * d_nfeatures + j];
            }
        }
    }

    if (local_tid == 0)
    {
        s_delta = 0;
    }

    __syncthreads();

    if (tid < d_npoints)
    {

        /* ========== find_nearest_point function start ========== */
        min_dist = FLT_MAX;
        for (i = 0; i < d_nclusters; i++)
        {

            /* ========== euclid_dist_2 function start ========== */
            dist = 0;

            for (j = 0; j < d_nfeatures; j++)
            {
                aux1 = d_feature[tid * d_nfeatures + j];
                aux2 = s_clusters[i * d_nfeatures + j];

                dist += (aux1 - aux2) * (aux1 - aux2);
            }

            /* ========== euclid_dist_2 function end ========== */

            if (dist < min_dist)
            {
                min_dist = dist;
                index = i;
            }
        }
        /* ========== find_nearest_point function end ========== */

        /* if membership changes, increase delta by 1 */
        if (s_delta < d_threshold && d_membership[tid] != index)
            atomicAdd(&s_delta, 1.0f);

        d_membership[tid] = index;
    }

    __syncthreads();

    // Store in main memory
    if (local_tid == 0 && *d_delta < d_threshold)
    {
        atomicAdd(d_delta, s_delta);
    }
}

// Kernel to sum cluster centers
__global__ void sum_clusters(float *d_feature, int *d_membership, float *d_new_centers, int *d_new_centers_len)
{
    int point = blockIdx.x * blockDim.x + threadIdx.x;
    int local_point = threadIdx.x;
    int feature = blockIdx.y * blockDim.y + threadIdx.y;
    int local_feature = threadIdx.y;
    int index;
    float *s_new_centers;
    int *s_new_centers_len;

    extern __shared__ char shared[];

    s_new_centers = (float *)shared;
    s_new_centers_len = (int *)(shared + d_nclusters * d_nfeatures * sizeof(float));

    // Init shared memory

    // Only nclusters
    if (local_feature == 0 && local_point < d_nclusters)
        s_new_centers_len[local_point] = 0;

    if (local_point < d_nclusters && local_feature < d_nfeatures)
        s_new_centers[local_point * d_nfeatures + local_feature] = 0.0f;

    __syncthreads();

    if (point < d_npoints && feature < d_nfeatures)
    {
        index = d_membership[point];

        // Only 1 thread per point
        if (feature == 0)
            atomicAdd(s_new_centers_len + index, 1);

        atomicAdd(s_new_centers + index * d_nfeatures + feature, d_feature[point * d_nfeatures + feature]);
    }

    __syncthreads();

    // Store in main memory
    if (local_feature == 0 && local_point < d_nclusters)
        atomicAdd(&d_new_centers_len[local_point], s_new_centers_len[local_point]);

    if (local_point < d_nclusters && local_feature < d_nfeatures)
        atomicAdd(&d_new_centers[local_point * d_nfeatures + local_feature], s_new_centers[local_point * d_nfeatures + local_feature]);
}

// Kernel to divide each new cluster center
__global__ void divide_clusters(float *d_clusters, float *d_new_centers, int *d_new_centers_len)
{
    int cluster = blockIdx.y * blockDim.y + threadIdx.y;
    int feature = blockIdx.x * blockDim.x + threadIdx.x;
    int len;

    extern __shared__ int s_new_centers_len[];

    if (feature == 0)
    {
        s_new_centers_len[cluster] = d_new_centers_len[cluster];
    }

    __syncthreads();

    if (cluster < d_nclusters && feature < d_nfeatures)
    {
        len = s_new_centers_len[cluster];
        if (len > 0)
        {
            d_clusters[cluster * d_nfeatures + feature] = d_new_centers[cluster * d_nfeatures + feature] / len;
        }
    }
}

/*----< kmeans_clustering() >---------------------------------------------*/
// To solve linking problems
extern "C" float **kmeans_clustering(float **feature, /* in: [npoints][nfeatures] */
                                     int nfeatures,
                                     int npoints,
                                     int nclusters,
                                     float threshold,
                                     int *membership) /* out: [npoints] */
{

    int i;
    float delta;
    float **clusters; /* out: [nclusters][nfeatures] */
    int optimized = 0;

    /* =============== Device vars =============== */
    int *d_membership;
    int *d_new_centers_len; /* [nclusters]: no. of points in each cluster */
    float *d_feature;
    float *d_delta;
    float *d_clusters;    /* out: [nclusters][nfeatures] */
    float *d_new_centers; /* [nclusters][nfeatures] */

    if (config_check(npoints, nfeatures, nclusters))
    {
        printf("Configuration of (npoints, nfeatures, nclusters) not currently supported (exceeds block and thread limits)\n");
        exit(1);
    }
    dim3 clusters_gridDist(1, 1, 1);
    dim3 clusters_blockDist(nfeatures, nclusters, 1);

    dim3 points_gridDist(updiv(TOTAL_THREAD_LIMIT / nfeatures, npoints), 1, 1);
    dim3 points_blockDist(TOTAL_THREAD_LIMIT / nfeatures, nfeatures, 1);

    checkError(cudaMalloc((void **)&d_membership, npoints * sizeof(int)));
    checkError(cudaMalloc((void **)&d_new_centers_len, nclusters * sizeof(int)));
    checkError(cudaMalloc((void **)&d_feature, npoints * nfeatures * sizeof(float)));
    checkError(cudaMalloc((void **)&d_delta, sizeof(float)));
    checkError(cudaMalloc((void **)&d_clusters, nclusters * nfeatures * sizeof(float)));
    checkError(cudaMalloc((void **)&d_new_centers, nclusters * nfeatures * sizeof(float)));

    cudaMemcpy2DAsync(d_feature, nfeatures * sizeof(float), feature[0], nfeatures * sizeof(float), nfeatures * sizeof(float), npoints, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbolAsync(d_nfeatures, &nfeatures, sizeof(int));
    cudaMemcpyToSymbolAsync(d_npoints, &npoints, sizeof(int));
    cudaMemcpyToSymbolAsync(d_nclusters, &nclusters, sizeof(int));
    cudaMemcpyToSymbolAsync(d_threshold, &threshold, sizeof(float));

    /* =============== allocate space for returning variable clusters[] =============== */
    clusters = (float **)malloc(nclusters * sizeof(float *));
    clusters[0] = (float *)malloc(nclusters * nfeatures * sizeof(float));

    if (clusters == NULL || clusters[0] == NULL)
    {
        printf("Couldn't allocate clusters, exiting.");
        exit(1);
    }

    for (i = 1; i < nclusters; i++)
        clusters[i] = clusters[i - 1] + nfeatures;

    checkError(cudaDeviceSynchronize()); // Check for errors in the Async calls

    /* =============== initialization  =============== */

    init_cluster_centers<<<clusters_gridDist, clusters_blockDist>>>(d_clusters, d_feature);
    checkError(cudaGetLastError());

    init_membership<<<updiv(THREADS_PER_BLOCK, npoints), THREADS_PER_BLOCK>>>(d_membership);
    checkError(cudaGetLastError());

    if (THREADS_PER_BLOCK >= nfeatures * nclusters)
        optimized = 1;
    /* =============== Main computation part  =============== */
    do
    {
        /* =============== Set auxiliary data structures to 0  =============== */
        reset_aux_data<<<clusters_gridDist, clusters_blockDist>>>(d_new_centers_len, d_new_centers, d_delta);
        checkError(cudaGetLastError());

        /* =============== assign membership to each point  =============== */
        assign_membership<<<updiv(THREADS_PER_BLOCK, npoints), THREADS_PER_BLOCK, nclusters * nfeatures * sizeof(float)>>>(d_feature, d_clusters, d_membership, d_new_centers, d_new_centers_len, d_delta, optimized);
        checkError(cudaGetLastError());

        /* =============== calculate the centers (average)  =============== */
        sum_clusters<<<points_gridDist, points_blockDist, nclusters * nfeatures * sizeof(float) + nclusters * sizeof(int)>>>(d_feature, d_membership, d_new_centers, d_new_centers_len);
        checkError(cudaGetLastError());
        divide_clusters<<<clusters_gridDist, clusters_blockDist, nclusters * sizeof(int)>>>(d_clusters, d_new_centers, d_new_centers_len);
        checkError(cudaGetLastError());

        checkError(cudaMemcpy(&delta, d_delta, sizeof(float), cudaMemcpyDeviceToHost)); // Also returns errors in the kernels

    } while (delta > threshold);

    /* =============== copy final results to host =============== */
    for (i = 0; i < nclusters; i++)
    {
        cudaMemcpyAsync(clusters[i], d_clusters + i * nfeatures, nfeatures * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaMemcpyAsync(membership, d_membership, npoints * sizeof(int), cudaMemcpyDeviceToHost);

    checkError(cudaDeviceSynchronize()); // Check for errors in the Async calls

    /* =============== free memory =============== */
    checkError(cudaFree(d_membership));
    checkError(cudaFree(d_new_centers_len));
    checkError(cudaFree(d_feature));
    checkError(cudaFree(d_delta));
    checkError(cudaFree(d_clusters));
    checkError(cudaFree(d_new_centers));

    return clusters;
}
