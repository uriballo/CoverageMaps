#include "FeedbackExpansionK.cuh"
#include <iostream>

__global__ void euclideanExpansion(cudaTextureObject_t domainTex, MapElement* coverageMap, bool* globalChanges, int rows, int cols, float radius) {
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

		int cellValue = tex2D<int>(domainTex, tidX, tidY);
		if (tidX < cols && tidY < rows && cellValue > -1) {
			bool updated =  listenUpdates(domainTex, coverageMap, tidX, tidY, rows, cols, radius);

			if (updated)
				blockChanges = true;
		}

		__syncthreads();

		if (blockChanges && responsible)
			*globalChanges = true;

	} while (blockChanges);
	
}

__device__ bool listenUpdates(cudaTextureObject_t domainTex, MapElement* coverageMap, int tidX, int tidY, int rows, int cols, float radius) {
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

			if (inBounds && neighIndex != pointIndex && coverageMap[neighIndex].predecessor != -1) {
				int cellValue = tex2D<int>(domainTex, nx, ny);
				if (cellValue > -1) {
					updates = updates || checkNeighInfo(domainTex, coverageMap, pointInfo, coverageMap[neighIndex], predPredInfo, pointIndex, neighIndex, rows, cols, radius);
				}
			}
		}
	}

	if (updates)
		coverageMap[pointIndex] = pointInfo;

	return updates;
}

__device__ bool checkNeighInfo(cudaTextureObject_t domainTex, MapElement* coverageMap, MapElement& pointInfo, MapElement neighInfo, MapElement predPredInfo, int pointIndex, int neighIndex, int rows, int cols, float radius) {
	bool expanded = false;

//	if (canUpdateInfo(domainTex, coverageMap, pointIndex, neighIndex, pointInfo.predecessor, predPredInfo.predecessor, rows, cols)) {
		float currentDistance = pointInfo.distance;
		float tentativeDistance = neighInfo.distance + indexDistance(pointIndex, neighIndex, cols);

		int predecessor = suitablePredecessor(domainTex, neighInfo.predecessor, neighIndex, cols);

	//	bool similarEnough = abs(currentDistance - tentativeDistance) < (sqrtf(2) - 1) * 0.0001;

		float distToPredecessor = indexDistance(pointIndex, pointInfo.predecessor, cols);
		float distToPotentialPred = indexDistance(pointIndex, predecessor, cols);

		if (/*(similarEnough && distToPotentialPred < distToPredecessor)
			||*/ (/*!similarEnough &&*/ tentativeDistance < currentDistance)) {
			
			int predX, predY;
			indexToCoords(predecessor, predX, predY, cols);

			int currentX, currentY;
			indexToCoords(pointIndex, currentX, currentY, cols);

			if (!visibilityTest(domainTex, rows, cols, predX, predY, currentX, currentY)) {
				return false;
			}

			MapElement newInfo{ tentativeDistance, predecessor, neighInfo.source };

			pointInfo = newInfo;

			expanded = true;
		}
//	}

	return expanded;
}

__device__ bool canUpdateInfo(cudaTextureObject_t domainTex, MapElement* coverageMap, int pointIndex, int neighIndex, int firstPredecessor, int secondPredecessor, int rows, int cols) {
	bool canUpdate = false;

	int predX, predY;
	indexToCoords(firstPredecessor, predX, predY, cols);

	int cellValue = tex2D<int>(domainTex, predX, predY);

	if (firstPredecessor == -1 || cellValue == 0 /*!isCorner(boundary, firstPredecessor, rows, cols)*/) {
		canUpdate = true;
	}
	else if (coverageMap[pointIndex].source == coverageMap[neighIndex].source) {
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
	else {
		canUpdate = true;
	}

	return canUpdate;
}

__device__ int suitablePredecessor(cudaTextureObject_t domainTex, int predecessorIndex, int neighIndex, int cols) {
	int neighX, neighY;
	indexToCoords(neighIndex, neighX, neighY, cols);

	int cellValue = tex2D<int>(domainTex, neighX, neighY);

	if (cellValue == 0)
		return predecessorIndex;
	else
		return neighIndex;
}

__global__ void EEDT(MapElement* coverageMap, bool* globalChanges, int rows, int cols, float radius) {
	// Get the ID of the thread

	int tidX, tidY;
	get2DThreadId(tidX, tidY);
	int tid = coordsToIndex(tidX, tidY, cols);

	if (tidX >= cols || tidY >= rows)
		return;

	MapElement pointInfo = coverageMap[tid];

	if (pointInfo.distance > 0 && pointInfo.distance < (radius + FLT_MIN) ) {
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

__global__ void evalCoverage(cudaTextureObject_t domainTex, MapElement* coverageMap, int rows, int cols, int* interiorPoints, int* coveredPoints) {
	int tidX, tidY;

	get2DThreadId(tidX, tidY);

	if (tidY < rows && tidX < cols) {
		bool interiorPoint = tex2D<int>(domainTex, tidX, tidY) > -1;

		int predecessor = coverageMap[coordsToIndex(tidX, tidY, cols)].predecessor;

		if (interiorPoint) {
			atomicAdd(interiorPoints, 1);
			if (predecessor != -1) {
				atomicAdd(coveredPoints, 1);
			}
		}
	}
}


__device__  bool visibilityTest(cudaTextureObject_t domainTex, int rows, int cols, int oX, int oY, int gX, int gY) {
	int dx = abs(gX - oX);
	int dy = abs(gY - oY);
	int sx = (oX < gX) ? 1 : -1;
	int sy = (oY < gY) ? 1 : -1;
	int err = dx - dy;

	while (true) {
		int cellValue = tex2D<int>(domainTex, oX, oY);
		if (cellValue == -1)
			return false;

		if (oX == gX && oY == gY)
			break;

		int e2 = 2 * err;

		if (e2 > -dy) {
			err -= dy;
			oX += sx;
		}

		if (e2 < dx) {
			err += dx;
			oY += sy;
		}

		if (oX >= cols || oX < 0 || oY >= rows || oY < 0)
			return false;
	}

	return true;
}