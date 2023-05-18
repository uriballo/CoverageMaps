#pragma once

//#ifndef FEEDBACK_EXP_K_CUH
//#define FEEDBACK_EXP_K_CUH

#include "CUDACommon.cuh"
#include <iostream>

__global__ void euclideanExpansion(const int* boundary, MapElement* coverageMap, bool* globalChanges, int rows, int cols, float radius);

__device__ bool listenUpdates(const int* boundary, MapElement* coverageMap, int tidX, int tidY, int rows, int cols, float radius);

__device__ bool checkNeighInfo(const int* boundary, MapElement* coverageMap, MapElement& pointInfo, MapElement neighInfo, MapElement predPredInfo, int pointIndex, int neighIndex, int rows, int cols, float radius);

__device__ bool canUpdateInfo(const int* boundary, MapElement* coverageMap, int point, int neigh, int firstPredecessor, int secondPredecessor, int rows, int cols);

__device__ int suitablePredecessor(const int* boundary, int predecessorIndex, int pointIndex, int neighIndex, int rows, int cols, float radius);

__global__ void EEDT(const int* boundary, MapElement* coverageMap, bool* globalChanges, int rows, int cols, float radius);

__global__ void initCoverageMap(MapElement* coverageMap, float initRadius, int initPredecessor, int* servicesDistribution, int numServices, int numElements, int cols);

//#endif
