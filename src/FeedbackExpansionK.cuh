#pragma once

#ifndef FEEDBACK_EXP_K_CUH
#define FEEDBACK_EXP_K_CUH

#include "CUDACommon.cuh"

__global__ void euclideanExpansion(const bool* boundary, CUDAPair<float, int>* coverageMap, bool* globalChanges, int rows, int cols, float radius);

__device__ bool listenUpdates(const bool* boundary, CUDAPair<float, int>* coverageMap, int tidX, int tidY, int rows, int cols, float radius);

__device__ bool checkNeighInfo(const bool* boundary, CUDAPair<float, int>* coverageMap, CUDAPair<float, int>& pointInfo, CUDAPair<float, int> neighInfo, CUDAPair<float, int> predPredInfo, int pointIndex, int neighIndex, int rows, int cols, float radius);

__device__ bool canUpdateInfo(const bool* boundary, CUDAPair<float, int>* coverageMap, int point, int neigh, int firstPredecessor, int secondPredecessor, int rows, int cols);

__device__ int suitablePredecessor(const bool* boundary, int predecessorIndex, int pointIndex, int neighIndex, int rows, int cols, float radius);

__global__ void EEDT(const bool* boundary, CUDAPair<float, int>* coverageMap, bool* globalChanges, int rows, int cols, float radius);

__global__ void EEDT_(const bool* boundary, CUDAPair<float, int>* coverageMap, bool* globalChanges, int rows, int cols, float radius);

#endif
