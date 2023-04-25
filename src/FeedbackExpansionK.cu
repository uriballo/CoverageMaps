#include "FeedbackExpansionK.cuh"
#include <iostream>

__global__ void euclideanExpansion(const bool* boundary, CUDAPair<float, int>* coverageMap, bool* globalChanges, int rows, int cols, float radius) {
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
			bool updated = listenUpdates(boundary, coverageMap, tid, rows, cols, radius);

			if (updated)
				blockChanges = true;
		}

		__syncthreads();

		if (blockChanges && responsible)
			*globalChanges = true;

	} while (blockChanges);

}

__device__ bool listenUpdates(const bool* boundary, CUDAPair<float, int>* coverageMap, int pointIndex, int rows, int cols, float radius) {
	bool updates = false;

	int ix = 0, iy = 0;
	indexToCoords(pointIndex, ix, iy, cols);

	CUDAPair<float, int> pointInfo = coverageMap[pointIndex];

	for (int dx = -1; dx < 2; dx++) {
		for (int dy = -1; dy < 2; dy++) {
			int nx = ix + dx;
			int ny = iy + dy;

			bool inBounds = nx < cols && ny < rows && nx >= 0 && ny >= 0;

			int neighIndex = coordsToIndex(nx, ny, cols);

			if (inBounds && neighIndex != pointIndex && !boundary[neighIndex]) {
				updates = updates || checkNeighInfo(boundary, coverageMap, pointInfo, coverageMap[neighIndex], pointIndex, neighIndex, rows, cols, radius);
			}
		}
	}

	if (updates)
		coverageMap[pointIndex] = pointInfo;

	return updates;
}

__device__ bool checkNeighInfo(const bool* boundary, CUDAPair<float, int>* coverageMap, CUDAPair<float, int>& pointInfo, CUDAPair<float, int> neighInfo, int pointIndex, int neighIndex, int rows, int cols, float radius) {
	bool expanded = false;

	if (neighInfo.second == -1)
		return expanded;

	if (canUpdateInfo(boundary, coverageMap, pointIndex, neighIndex, pointInfo.second, coverageMap[pointInfo.second].second, rows, cols)) {
		float currentDistance = pointInfo.first;

		int predecessor = suitablePredecessor(boundary, neighInfo.second, pointIndex, neighIndex, rows, cols, radius);

		float distanceToNeigh = indexDistance(pointIndex, neighIndex, cols);
		float tentativeDistance = neighInfo.first + distanceToNeigh;

		float totalDistance = stepDistance(coverageMap, neighIndex, cols) + distanceToNeigh;


		if (/*totalDistance < radius + FLT_MIN &&*/ tentativeDistance < currentDistance) {
			//printf("> %d -> %d\n", currentDistance, tentativeDistance);
			//int predecessor = suitablePredecessor(boundary, neighInfo.second, pointIndex, neighIndex, rows, cols, radius);
			// 
			//float distanceToPredecessor = indexDistance(pointIndex, predecessor, cols);
			//float dd = computeDistance(coverageMap, predecessor, cols);

			//CUDAPair<float, int> newInfo{ distanceToPredecessor + dd, predecessor };
			CUDAPair<float, int> newInfo{ tentativeDistance, predecessor };

			pointInfo = newInfo;

			expanded = true;
		}
	}

	return expanded;
}

__device__ bool canUpdateInfo(const bool* boundary, CUDAPair<float, int>* coverageMap, int pointIndex, int neighIndex, int firstPredecessor, int secondPredecessor, int rows, int cols) {
	if (firstPredecessor == -1 || !isNearBoundary(boundary, firstPredecessor, rows)) {
		return true;
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

		if (sgn(det1) == sgn(det2)) {
			int det3 = det(pX - preX, pY - preY, nX - preX, nY - preY);

			if (sgn(det1) == sgn(det3)) {
				return true;
			}
		}
		else {
			return false;
		}
	}
}

// whichSubgoal -> suitablePredecessor
__device__ int suitablePredecessor(const bool* boundary, int predecessorIndex, int pointIndex, int neighIndex, int rows, int cols, float radius) {
	bool neighIsNearBoundary = isNearBoundary(boundary, neighIndex, rows);
	bool pointIsNearBoundary = isNearBoundary(boundary, pointIndex, rows);

	int pX, pY;
	indexToCoords(pointIndex, pX, pY, cols);

	int nX, nY;
	indexToCoords(neighIndex, nX, nY, cols);

	int preX, preY;
	indexToCoords(predecessorIndex, preX, preY, cols);

	if (!neighIsNearBoundary || !pointIsNearBoundary)
		return predecessorIndex;
	else if ((pX == nX && nX == preX) || (pY == nY && nY == preY))
		return predecessorIndex;
	else
		return neighIndex;
}

__global__ void EEDT(const bool* boundary, CUDAPair<float, int>* coverageMap, bool* globalChanges, int rows, int cols, float radius) {
	// Get the ID of the thread
	int tid = getThreadId();
	int dims = rows * cols;

	if (tid >= dims)
		return;

	CUDAPair<float, int> pointInfo = coverageMap[tid];

	if (!boundary[tid] && pointInfo.first < (radius + FLT_MIN) && pointInfo.second != -1) {
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
