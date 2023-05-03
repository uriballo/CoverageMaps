#pragma once

#ifndef EUCLIDEAN_EXP_K_CUH
#define EUCLIDEAN_EXP_K_CUH

#include "CUDACommon.cuh"

__global__ void euclideanExpansionKernel(const bool* boundary, float* distances, const int* sources, bool* globalChanges, int rows, int cols);

__device__ bool scanWindow(const bool* boundary, float* distances, const int* sources, const int pixelIndex, int rows, int cols);

__device__ bool updateDistances(float* distances, int pointIndex, int neighIndex, bool diag);



#endif // EUCLIDEAN_EXP_KERNELS_CUH