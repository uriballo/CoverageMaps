#include "FeedbackExpansionK.cuh"
#include <iostream>

__global__ void euclideanExpansion(const int* boundary, MapElement* coverageMap, bool* globalChanges, int rows, int cols, float radius) {
	int tidX, tidY;
	get2DThreadId(tidX, tidY);
	
	int index = coordsToIndex(tidX, tidY, cols);
	bool responsible = threadIdx.x == 0 && threadIdx.y == 0;

	__shared__ bool blockChanges;

	do {
		if (responsible)
			// Set the value of blockIsActive to false at the start of each iteration
			blockChanges = false;

		__syncthreads();

		int tid = coordsToIndex(tidX, tidY, cols);

		if (tidX < cols && tidY < rows && boundary[tid] > -1) {
			bool updated = listenUpdates(boundary, coverageMap, tidX, tidY, rows, cols, radius);

			if (updated)
				blockChanges = true;
		}

		__syncthreads();

		if (blockChanges && responsible)
			*globalChanges = true;

	} while (blockChanges);
	
}

__device__ bool listenUpdates(const int* boundary, MapElement* coverageMap, int tidX, int tidY, int rows, int cols, float radius) {
	bool updates = false;

	int pointIndex = coordsToIndex(tidX, tidY, cols);

	MapElement pointInfo = coverageMap[pointIndex];

	// TODO: problema concu
	MapElement predPredInfo = coverageMap[pointInfo.predecessor];

	for (int dx = -1; dx < 2; dx++) {
		for (int dy = -1; dy < 2; dy++) {
			int nx = tidX + dx;
			int ny = tidY + dy;

			bool inBounds = nx < cols && ny < rows && nx >= 0 && ny >= 0;

			int neighIndex = coordsToIndex(nx, ny, cols);

			if (inBounds && neighIndex != pointIndex) {

				if (boundary[neighIndex] > -1) {
					updates = updates || checkNeighInfo(boundary, coverageMap, pointInfo, coverageMap[neighIndex], predPredInfo, pointIndex, neighIndex, rows, cols, radius);
				}
			}
		}
	}

	if (updates)
		coverageMap[pointIndex] = pointInfo;

	return updates;
}

__device__ bool checkNeighInfo(const int* boundary, MapElement* coverageMap, MapElement& pointInfo, MapElement neighInfo, MapElement predPredInfo, int pointIndex, int neighIndex, int rows, int cols, float radius) {
	bool expanded = false;

	if (neighInfo.predecessor == -1)
		return expanded;

	if (canUpdateInfo(boundary, coverageMap, pointIndex, neighIndex, pointInfo.predecessor, predPredInfo.predecessor, rows, cols)) {
		float currentDistance = pointInfo.distance;

		float distanceToNeigh = indexDistance(pointIndex, neighIndex, cols);
		float tentativeDistance = neighInfo.distance + distanceToNeigh;

		int predecessor = suitablePredecessor(boundary, neighInfo.predecessor, pointIndex, neighIndex, rows, cols, radius);

		float threshold = (sqrtf(2) - 1) * 0.49;

		float diff = abs(currentDistance - tentativeDistance);
		bool similarEnough = diff < threshold;

		float distToPredecessor = indexDistance(pointIndex, pointInfo.predecessor, cols);
		float distToPotentialPred = indexDistance(pointIndex, predecessor, cols);

		if ((similarEnough && distToPotentialPred < distToPredecessor)
			|| (!similarEnough && tentativeDistance < currentDistance)){
			
			// TEST VISIBILITAT AQUI
			
		//	if ((similarEnough && distToPotentialPred < distToPredecessor) && pointInfo.second != -1) {
		//	  printf("> Tentative: %f\n\t, Current %f\n\t, Tentative - Current Distance is: %f\n\tneigh: %d\n\t, pred: %d\n\t, potentialPred: %d\n\t, isPredCorner: %d\n\t, isPotPredCorner: %d\n\t, pixelIndex %d\n\n", tentativeDistance, currentDistance,  diff , neighIndex, pointInfo.second, predecessor, isCorner(boundary, pointInfo.second, rows, cols), isCorner(boundary, predecessor, rows, cols ), pointIndex);
		//	}

			MapElement newInfo{ tentativeDistance, predecessor, neighInfo.source };

			pointInfo = newInfo;

			expanded = true;
		}
	}

	return expanded;
}

__device__ bool canUpdateInfo(const int* boundary, MapElement* coverageMap, int pointIndex, int neighIndex, int firstPredecessor, int secondPredecessor, int rows, int cols) {
	bool canUpdate = false;

	if (firstPredecessor == -1 || boundary[firstPredecessor] == 0 /*!isCorner(boundary, firstPredecessor, rows, cols)*/) {
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

		if (sgn(det1) == sgn(det2)) {

			int det3 = det(pX - preX, pY - preY, nX - preX, nY - preY);
			
			if (sgn(det2) == sgn(det3)) {
				canUpdate = true;
			}
		}
	}

	return canUpdate;
}

__device__ int suitablePredecessor(const int* boundary, int predecessorIndex, int pointIndex, int neighIndex, int rows, int cols, float radius) {
	if (boundary[neighIndex] == 0)
		return predecessorIndex;
	else
		return neighIndex;
}

__global__ void EEDT(const int* boundary, MapElement* coverageMap, bool* globalChanges, int rows, int cols, float radius) {
	// Get the ID of the thread

	int tidX, tidY;
	get2DThreadId(tidX, tidY);
	int tid = coordsToIndex(tidX, tidY, cols);

	if (tidX >= cols || tidY >= rows)
		return;

	MapElement pointInfo = coverageMap[tid];

	if (boundary[tid] > -1 && pointInfo.distance > 0 && pointInfo.distance < (radius + FLT_MIN) ) {
		float exactDistance = computeDistance(coverageMap, tid, cols);

		if (exactDistance != -1 && exactDistance < pointInfo.distance) {
			coverageMap[tid].distance = exactDistance;
			*globalChanges = true;
		}
	}
}

__global__ void initCoverageMap(MapElement* coverageMap, float initRadius, int initPredecessor, int* servicesDistribution, int numServices, int numElements, int cols) {
	int tid = getThreadId();
	bool responsible = threadIdx.x == 0 && threadIdx.y == 0;

	if (tid < numElements) {
		MapElement currentCell;

		currentCell.distance = initRadius;
		currentCell.predecessor = initPredecessor;
		currentCell.source = initPredecessor;

		coverageMap[tid] = currentCell;
	}

	__syncthreads();

	if (responsible) {
		for (int i = 0; i < 2 * numServices; i += 2) {
			int serviceIndex = coordsToIndex(servicesDistribution[i], servicesDistribution[i + 1], cols);

			coverageMap[serviceIndex].distance = 0;
			coverageMap[serviceIndex].predecessor = serviceIndex;
			coverageMap[serviceIndex].source = serviceIndex;
		}
	}
}
