#include "FeedbackExpansionK.cuh"
#include <iostream>

__global__ void euclideanExpansion(const bool* boundary, CUDAPair<float, int>* coverageMap, bool* globalChanges, int rows, int cols, float radius) {
	// Get the ID of the thread
	//int tid = getThreadId();

	int tidX, tidY;
	get2DThreadId(tidX, tidY);

	
	int index = coordsToIndex(tidX, tidY, cols);
/*
	if (boundary[index]) {
		coverageMap[index].first = -1;
		coverageMap[index].second = -1;
	}
	else if (isCorner(boundary, index, rows, cols)) {
		coverageMap[index].first = 999;
		coverageMap[index].second = index;
	}	
	else {
		coverageMap[index].first = 0;
		coverageMap[index].second = index;
	}
	return;
	*/
	bool responsible = threadIdx.x == 0 && threadIdx.y == 0;

	__shared__ bool blockChanges;

	do {
		if (responsible)
			// Set the value of blockIsActive to false at the start of each iteration
			blockChanges = false;

		__syncthreads();

		int tid = coordsToIndex(tidX, tidY, cols);

		if (tidX < cols && tidY < rows && !boundary[tid]) {
			bool updated = listenUpdates(boundary, coverageMap, tidX, tidY, rows, cols, radius);

			if (updated)
				blockChanges = true;
		}

		__syncthreads();

		if (blockChanges && responsible)
			*globalChanges = true;

	} while (blockChanges);
	
}

__device__ bool listenUpdates(const bool* boundary, CUDAPair<float, int>* coverageMap, int tidX, int tidY, int rows, int cols, float radius) {
	bool updates = false;

	int pointIndex = coordsToIndex(tidX, tidY, cols);

	CUDAPair<float, int> pointInfo = coverageMap[pointIndex];

	// TODO: problema concu
	CUDAPair<float, int> predPredInfo = coverageMap[pointInfo.second];

	for (int dx = -1; dx < 2; dx++) {
		for (int dy = -1; dy < 2; dy++) {
			int nx = tidX + dx;
			int ny = tidY + dy;

			bool inBounds = nx < cols && ny < rows && nx >= 0 && ny >= 0;

			int neighIndex = coordsToIndex(nx, ny, cols);

			if (inBounds && neighIndex != pointIndex) {

				if (!boundary[neighIndex]) {
					updates = updates || checkNeighInfo(boundary, coverageMap, pointInfo, coverageMap[neighIndex], predPredInfo, pointIndex, neighIndex, rows, cols, radius);
				}
			}
		}
	}

	if (updates)
		coverageMap[pointIndex] = pointInfo;

	return updates;
}

__device__ bool checkNeighInfo(const bool* boundary, CUDAPair<float, int>* coverageMap, CUDAPair<float, int>& pointInfo, CUDAPair<float, int> neighInfo, CUDAPair<float, int> predPredInfo, int pointIndex, int neighIndex, int rows, int cols, float radius) {
	bool expanded = false;

	if (neighInfo.second == -1)
		return expanded;

//	if (canUpdateInfo(boundary, coverageMap, pointIndex, neighIndex, pointInfo.second, predPredInfo.second, rows, cols)) {
		float currentDistance = pointInfo.first;

		float distanceToNeigh = indexDistance(pointIndex, neighIndex, cols);
		float tentativeDistance = neighInfo.first + distanceToNeigh;

		int predecessor = suitablePredecessor(boundary, neighInfo.second, pointIndex, neighIndex, rows, cols, radius);

		bool similarEnough = abs(tentativeDistance - currentDistance) < 0.41;

		float distToPredecessor = indexDistance(pointIndex, pointInfo.second, cols);
		float distToPotentialPred = indexDistance(pointIndex, predecessor, cols);

		if ((similarEnough && distToPotentialPred < distToPredecessor)
			|| (!similarEnough && tentativeDistance < currentDistance)){

			CUDAPair<float, int> newInfo{ tentativeDistance, predecessor };

			pointInfo = newInfo;

			expanded = true;
		}
//	}

	return expanded;
}

__device__ bool canUpdateInfo(const bool* boundary, CUDAPair<float, int>* coverageMap, int pointIndex, int neighIndex, int firstPredecessor, int secondPredecessor, int rows, int cols) {
	bool canUpdate = false;

	if (firstPredecessor == -1 || !isNearBoundary(boundary, firstPredecessor, rows, cols)) {
		canUpdate = true;
	}
	else {
		int pX, pY;
		indexToCoords(pointIndex, pX, pY, cols);

		int nX, nY;
		indexToCoords(neighIndex, nX, nY, cols);

		int preX, preY;
		indexToCoords(firstPredecessor, preX, preY, cols);

		int preX2, preY2;
		indexToCoords(secondPredecessor, preX2, preY2, cols);

		int det1 = det(preX - preX2, preY - preY2, pX - preX2, pY - preY2);
		int det2 = det(preX - preX2, preY - preY2, nX - preX2, nY - preY2);

		if (pointIndex == 397595) {
			printf("\n\n-----\nDet1: %d (sgn: %d)\nDet2: %d (sgn: %d)\n", det1, sgn(det1), det2, sgn(det2));
		}
		if (sgn(det1) == sgn(det2)) {
			int det3 = det(pX - preX, pY - preY, nX - preX, nY - preY);
			if (pointIndex == 397595) {
				printf("Det3: %d (sgn: %d)\n-----\n", det3, sgn(det3));
			}
			if (sgn(det2) == sgn(det3)) {
				canUpdate = true;
			}
		}
	}

	return canUpdate;
}

// whichSubgoal -> suitablePredecessor
__device__ int suitablePredecessor(const bool* boundary, int predecessorIndex, int pointIndex, int neighIndex, int rows, int cols, float radius) {
//	bool neighIsNearBoundary = isNearBoundary(boundary, neighIndex, rows, cols);
//	bool pointIsNearBoundary = isNearBoundary(boundary, pointIndex, rows, cols);

//	int pX, pY;
//	indexToCoords(pointIndex, pX, pY, cols);

//	int nX, nY;
//	indexToCoords(neighIndex, nX, nY, cols);

//	int preX, preY;
//	indexToCoords(predecessorIndex, preX, preY, cols);

	/*
	if (!neighIsNearBoundary || !pointIsNearBoundary)
		return predecessorIndex;
	else if ((pX == nX && nX == preX) || (pY == nY && nY == preY))
		return predecessorIndex;
	else 
		return neighIndex;
	*/
	
	if (!isCorner(boundary, neighIndex, rows, cols))
		return predecessorIndex;
	else
		return neighIndex;
}

__global__ void EEDT(const bool* boundary, CUDAPair<float, int>* coverageMap, bool* globalChanges, int rows, int cols, float radius) {
	// Get the ID of the thread

	int tidX, tidY;
	get2DThreadId(tidX, tidY);
	int tid = coordsToIndex(tidX, tidY, cols);

	if (tidX >= cols || tidY >= rows)
		return;

	CUDAPair<float, int> pointInfo = coverageMap[tid];

	if (!boundary[tid] && pointInfo.first > 0 && pointInfo.first < (radius + FLT_MIN) ) {
		float exactDistance = computeDistance(coverageMap, tid, cols);

		if (exactDistance != -1 && exactDistance < pointInfo.first) {
			coverageMap[tid].first = exactDistance;
			*globalChanges = true;
		}
	}
}

__global__ void EEDT_(const bool* boundary, CUDAPair<float, int>* coverageMap, bool* globalChanges, int rows, int cols, float radius) {
	// Get the ID of the thread
	int tid = getThreadId();

	bool responsible = threadIdx.x == 0 && threadIdx.y == 0;

	__shared__ bool blockChanges;

	int dims = rows * cols;

	do {
		if (responsible)
			// Set the value of blockIsActive to false at the start of each iteration
			blockChanges = false;

		__syncthreads();

		if (tid < dims && !boundary[tid]) {
			CUDAPair<float, int> pointInfo = coverageMap[tid];

			float predecessorDistance = indexDistance(tid, pointInfo.second, cols);

			if (predecessorDistance < pointInfo.first) {
				coverageMap[tid].first = predecessorDistance;
				blockChanges = true;
			}
		}

		__syncthreads();

		if (blockChanges && responsible)
			*globalChanges = true;
	} while (blockChanges);
}
