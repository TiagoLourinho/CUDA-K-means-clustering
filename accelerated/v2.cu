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

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

extern double wtime(void);

__constant__ int d_nfeatures;
__constant__ int d_npoints;
__constant__ int d_nclusters;

/* ==================== Host util functions ==================== */

int choose_number_of_blocks(int threads_per_block, int N)
{
    return (N + threads_per_block - 1) / threads_per_block;
}

/* ==================== Device util functions ==================== */

__device__ int find_nearest_point(float *pt, /* [nfeatures] */
                                  int nfeatures,
                                  float *pts, /* [npts][nfeatures] */
                                  int npts)
{
    int index, i;
    float min_dist = FLT_MAX;

    /* find the cluster center id with min distance to pt */
    for (i = 0; i < npts; i++)
    {
        float dist;
        dist = euclid_dist_2(pt, pts + i * nfeatures, nfeatures); /* no need square root */
        if (dist < min_dist)
        {
            min_dist = dist;
            index = i;
        }
    }
    return (index);
}

__device__ float euclid_dist_2(float *pt1,
                               float *pt2,
                               int numdims)
{
    int i;
    float ans = 0.0;

    for (i = 0; i < numdims; i++)
        ans += (pt1[i] - pt2[i]) * (pt1[i] - pt2[i]);

    return (ans);
}

/* ==================== Init functions ==================== */

__global__ void init_cluster_centers(float *d_clusters, float *d_feature)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;

    if ((col < d_nclusters) && (row < d_nfeatures))
    {
            d_clusters[d_nclusters * row + col] = d_feature[d_nclusters * row + col];
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

/* ==================== Reset functions ==================== */

__global__ void reset_new_centers_len(int *d_new_centers_len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < d_nclusters)
    {
        d_new_centers_len[tid] = 0;
    }
}

__global__ void reset_new_centers(float *d_new_centers)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int N = d_nclusters * d_nfeatures;

    if (tid < N)
    {
        d_new_centers[tid] = 0.0;
    }
}

__global__ void reset_delta(float *d_delta)
{
    *(d_delta) = 0.0;
}

/* ==================== Main computational functions ==================== */

__global__ void assign_membership(float *d_feature, float *d_clusters, int *d_membership, float *d_new_centers, int *d_new_centers_len, float *d_delta)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int index, j;

    if (tid < d_npoints)
    {
        /* find the index of nestest cluster centers */
        index = find_nearest_point(d_feature + tid * d_nfeatures, d_nfeatures, d_clusters, d_nclusters);

        /* if membership changes, increase delta by 1 */
        if (*(d_membership + tid) != index)
            atomicAdd(d_delta, 1.0f);

        /* assign the membership to object i */
        *(d_membership + tid) = index;

        /* update new cluster centers : sum of objects located within */
        atomicAdd(d_new_centers_len + index, 1);

        for (j = 0; j < d_nfeatures; j++)
            atomicAdd(d_new_centers + index * d_nfeatures + j, *(d_feature + tid * d_nfeatures + j));
    }
}

__global__ void update_clusters(float *d_clusters, float *d_new_centers, int *d_new_centers_len)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;

    if (col < d_nclusters)
    {
        if (d_new_centers_len[col] > 0)
        {
                d_clusters[col * d_nfeatures + row] = d_new_centers[col * d_nfeatures + row] / d_new_centers_len[col];
        }
    }
}

/*----< kmeans_clustering() >---------------------------------------------*/
float **kmeans_clustering(float **feature, /* in: [npoints][nfeatures] */
                          int nfeatures,
                          int npoints,
                          int nclusters,
                          float threshold,
                          int *membership) /* out: [npoints] */
{

    int i, threads_per_block_clusters;
    float delta;
    float **clusters; /* out: [nclusters][nfeatures] */

    /* =============== Device vars =============== */
    int *d_membership;
    int *d_new_centers_len; /* [nclusters]: no. of points in each cluster */
    float *d_feature;
    float *d_delta;
    float *d_clusters;    /* out: [nclusters][nfeatures] */
    float *d_new_centers; /* [nclusters][nfeatures] */

    dim3 gridDist (choose_number_of_blocks(THREADS_PER_BLOCK, nclusters), nfeatures, 1);
    dim3 blockDist(THREADS_PER_BLOCK, 1, 1);

    cudaMalloc((void **)&d_membership, npoints * sizeof(int));
    cudaMalloc((void **)&d_new_centers_len, nclusters * sizeof(int));
    cudaMalloc((void **)&d_feature, npoints * nfeatures * sizeof(float));
    cudaMalloc((void **)&d_delta, sizeof(float));
    cudaMalloc((void **)&d_clusters, nclusters * nfeatures * sizeof(float));
    cudaMalloc((void **)&d_new_centers, nclusters * nfeatures * sizeof(float));

    cudaMemcpyToSymbol(d_nfeatures, &nfeatures, sizeof(int));
    cudaMemcpyToSymbol(d_npoints, &npoints, sizeof(int));
    cudaMemcpyToSymbol(d_nclusters, &nclusters, sizeof(int));

    for (i = 0; i < npoints; i++)
        cudaMemcpy(d_feature + i * nfeatures, feature[i], nfeatures * sizeof(float), cudaMemcpyHostToDevice);

    /* =============== allocate space for returning variable clusters[] =============== */
    clusters = (float **)malloc(nclusters * sizeof(float *));
    clusters[0] = (float *)malloc(nclusters * nfeatures * sizeof(float));
    for (i = 1; i < nclusters; i++)
        clusters[i] = clusters[i - 1] + nfeatures;

    /* =============== initialization  =============== */
    init_cluster_centers<<<gridDist, blockDist>>>(d_clusters, d_feature);
    //init_cluster_centers<<<choose_number_of_blocks(WARP_SIZE, nclusters), WARP_SIZE>>>(d_clusters, d_feature);

    init_membership<<<choose_number_of_blocks(THREADS_PER_BLOCK, npoints), THREADS_PER_BLOCK>>>(d_membership);

    if (nclusters < THREADS_PER_BLOCK) {
        threads_per_block_clusters = nclusters;
    }

    do
    {

        /* =============== reset vars =============== */
        reset_new_centers_len<<<choose_number_of_blocks(THREADS_PER_BLOCK, nclusters), threads_per_block_clusters>>>(d_new_centers_len);
        reset_new_centers<<<choose_number_of_blocks(THREADS_PER_BLOCK, npoints), THREADS_PER_BLOCK>>>(d_new_centers);
        reset_delta<<<1, 1>>>(d_delta);

        /* =============== assign membership =============== */
        assign_membership<<<choose_number_of_blocks(THREADS_PER_BLOCK, npoints), THREADS_PER_BLOCK>>>(d_feature, d_clusters, d_membership, d_new_centers, d_new_centers_len, d_delta);

        /* =============== replace old cluster centers with new_centers and update delta =============== */
        update_clusters<<<gridDist, blockDist>>>(d_clusters, d_new_centers, d_new_centers_len);

        /* =============== get delta =============== */
        cudaMemcpy(&delta, d_delta, sizeof(float), cudaMemcpyDeviceToHost);

    } while (delta > threshold);

    /* =============== copy final clusters to host =============== */
    for (i = 0; i < nclusters; i++)
    {
        cudaMemcpy(clusters[i], d_clusters + i * nfeatures, nfeatures * sizeof(float), cudaMemcpyDeviceToHost);
    }

    /* =============== free memory =============== */
    cudaFree(d_membership);
    cudaFree(d_new_centers_len);
    cudaFree(d_feature);
    cudaFree(d_delta);
    cudaFree(d_clusters);
    cudaFree(d_new_centers);

    return clusters;
}
