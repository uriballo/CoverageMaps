#pragma once

#ifndef FEEDBACK_EXP_K_CUH
#define FEEDBACK_EXP_K_CUH

#include "CUDACommon.cuh"
#include <iostream>

__global__ void euclideanExpansion(const bool* boundary, CUDAPair<float, int>* coverageMap, bool* globalChanges, int rows, int cols, float radius);

__device__ bool listenUpdates(const bool* boundary, CUDAPair<float, int>* coverageMap, int tidX, int tidY, int rows, int cols, float radius);

__device__ bool checkNeighInfo(const bool* boundary, CUDAPair<float, int>* coverageMap, CUDAPair<float, int>& pointInfo, CUDAPair<float, int> neighInfo, CUDAPair<float, int> predPredInfo, int pointIndex, int neighIndex, int rows, int cols, float radius);

__device__ bool canUpdateInfo(const bool* boundary, CUDAPair<float, int>* coverageMap, int point, int neigh, int firstPredecessor, int secondPredecessor, int rows, int cols);

__device__ int suitablePredecessor(const bool* boundary, int predecessorIndex, int pointIndex, int neighIndex, int rows, int cols, float radius);

__global__ void EEDT(const bool* boundary, CUDAPair<float, int>* coverageMap, bool* globalChanges, int rows, int cols, float radius);

__global__ void EEDT_(const bool* boundary, CUDAPair<float, int>* coverageMap, bool* globalChanges, int rows, int cols, float radius);

__global__ __inline__ void testKernel(const bool* domain, CUDAPair<float, int>* distanceMap, int rows, int cols, float radius) {
	// Get the ID of the thread

	int tidX, tidY;
	get2DThreadId(tidX, tidY);

	if (tidX < cols && tidY < rows) {
		int index = coordsToIndex(tidX, tidY, cols);

		if (domain[index]) {
			printf("> %d\n", index);
			distanceMap[index].second = 0;
		}
		else {
			int pX, pY;


			indexToCoords(434678, pX, pY, cols);


		//	if (visibilityTest(domain, rows, cols, pX, pY, tidX, tidY)) {
		//		distanceMap[index].second = 1;
		//		printf("> %d is visible\n", index);
		//	}
	//		else
				distanceMap[index].second = 0;
		}

	}
}

#endif
