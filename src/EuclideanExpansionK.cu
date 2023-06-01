#include "EuclideanExpansionK.cuh"
#include <iostream>

__global__ void pseudoEuclideanExpansion(cudaTextureObject_t domainTex, MapElement* coverageMap, bool* globalChanges, int rows, int cols) {
	int tidX, tidY;
	get2DThreadId(tidX, tidY);

	int index = coordsToIndex(tidX, tidY, cols);
	bool responsible = threadIdx.x == 0 && threadIdx.y == 0;

	__shared__ bool blockChanges;

	do {
		if (responsible)
			// Set the value of blockChanges to false at the start of each iteration.
			blockChanges = false;

		__syncthreads();

		int tid = coordsToIndex(tidX, tidY, cols);

		int cellValue = tex2D<int>(domainTex, tidX, tidY);
		if (tidX < cols && tidY < rows && cellValue > -1) {
			bool updated = scanWindow(domainTex, coverageMap, tid, rows, cols);

			if (updated)
				blockChanges = true;
		}

		__syncthreads();

		if (blockChanges && responsible)
			*globalChanges = true;

	} while (blockChanges);
}

__device__ bool scanWindow(cudaTextureObject_t domainTex, MapElement* coverageMap, const int pixelIndex, int rows, int cols) {
	int ix = 0, iy = 0;
	indexToCoords(pixelIndex, ix, iy, cols);

	bool updated = false;

	// Loop through all the neighboring pixels.
	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			int newX = ix + i;
			int newY = iy + j;

			bool xInBounds = newX < cols && newX >= 0;
			bool yInBounds = newY < rows && newY >= 0;
			bool inBounds = xInBounds && yInBounds;

			if (inBounds) {
				int neighIndex = coordsToIndex(newX, newY, cols);

				// Check if it's a wall.
				int cellValue = tex2D<int>(domainTex, newX, newY);
				bool isWall = cellValue == -1;

				if (!isWall && pixelIndex != neighIndex) {
					// Determine whether the neighboring pixel is diagonal.
					bool diag = abs(i) == abs(j);

					updated = updated || checkNeighInfo(coverageMap, pixelIndex, neighIndex, diag);
				}
			}
		}
	}
	return updated;
}

__device__ bool checkNeighInfo(MapElement* coverageMap, int pointIndex, int neighIndex, bool diag) {
	// Compute the distance between the current point and the neighboring point.
	float distanceBetween = diag ? diagStepDistance() : unitStepDistance();
	bool updated = false;

	float currentDistance = coverageMap[pointIndex].distance;
	float tentativeDistance = coverageMap[neighIndex].distance + distanceBetween;

	if (tentativeDistance < currentDistance) {
		coverageMap[pointIndex].distance = tentativeDistance;

		if (coverageMap[neighIndex].predecessor != -1)
			coverageMap[pointIndex].predecessor = coverageMap[neighIndex].predecessor;
		updated = true;
	}

	return updated;
}
