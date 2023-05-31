#pragma once

//#ifndef FEEDBACK_EXP_K_CUH
//#define FEEDBACK_EXP_K_CUH

#include "CUDACommon.cuh"
#include <iostream>
#include <cuda_texture_types.h>


__global__ void euclideanExpansion(cudaTextureObject_t domainTex, MapElement* coverageMap, bool* globalChanges, int rows, int cols, float radius);

__device__ bool listenUpdates(cudaTextureObject_t domainTex, MapElement* coverageMap, int tidX, int tidY, int rows, int cols, float radius);

__device__ bool checkNeighInfo(cudaTextureObject_t domainTex, MapElement* coverageMap, MapElement& pointInfo, MapElement neighInfo, MapElement predPredInfo, int pointIndex, int neighIndex, int rows, int cols, float radius);

__device__ bool canUpdateInfo(cudaTextureObject_t domainTex, MapElement* coverageMap, int point, int neigh, int firstPredecessor, int secondPredecessor, int rows, int cols);

__device__ int suitablePredecessor(cudaTextureObject_t domainTex, int predecessorIndex,int neighIndex, int cols);

__global__ void EEDT(MapElement* coverageMap, bool* globalChanges, int rows, int cols, float radius);

__global__ void initializeCoverageMap(MapElement* coverageMap, float initRadius, int initPredecessor, int* servicesDistribution, int numServices, int numElements, int cols);

__global__ void evalCoverage(cudaTextureObject_t domainTex, MapElement* coverageMap, int rows, int cols, int* interiorPoints, int* coveredPoints);

__device__ bool visibilityTest(cudaTextureObject_t domainTex, int rows, int cols, int oX, int oY, int gX, int gY);


//#endif
