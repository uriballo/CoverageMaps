#include "EuclideanExpansionK.cuh"
#include <iostream>

__global__ void euclideanExpansionKernel(const bool* boundary, float* distances, bool* globalChanges, int rows, int cols) {
	// Get the ID of the thread
	int tid = getThreadId();

	bool responsible = threadIdx.x == 0 && threadIdx.y == 0;

	int dims = rows * cols;

	__shared__ bool blockChanges;

	do {
		// Set the value of blockIsActive to false at the start of each iteration
		if (responsible)
			blockChanges = false;

		__syncthreads();
		if (tid < dims && !boundary[tid]) {
			if (scanWindow(boundary, distances, tid, rows, cols))
				blockChanges = true;
		}

		__syncthreads();

		if (blockChanges && responsible)
			*globalChanges = true;

	} while (blockChanges);
}

__device__ bool scanWindow(const bool* boundary, float* distances, const int pixelIndex, int rows, int cols) {
	// Convert the index of the current pixel to its x and y coordinates
	int ix = 0, iy = 0;
	indexToCoords(pixelIndex, ix, iy, cols);

	// Declare a boolean variable to keep track of whether this pixel was updated
	bool updated = false;

	// Loop through all the neighboring pixels
	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			int newX = ix + i;
			int newY = iy + j;

			bool xInBounds = newX < cols && newX >= 0;
			bool yInBounds = newY < rows && newY >= 0;
			bool inBounds = xInBounds && yInBounds;

			if (inBounds) {
				// Compute the neighbour index
				int neighIndex = coordsToIndex(newX, newY, cols);

				// Check if its a wall
				bool isWall = boundary[neighIndex];

				if (!isWall && pixelIndex != neighIndex) {
					// Determine whether the neighboring pixel is diagonal or not
					bool diag = abs(i) == abs(j);

					updated = updated || updateDistances(distances, pixelIndex, neighIndex, diag);
				}
			}
		}
	}
	return updated;
}

__device__ bool updateDistances(float* distances, int pointIndex, int neighIndex, bool diag) {
	// Compute the distance between the current point and the neighboring point
	float distanceBetween = diag ? diagStepDistance() : unitStepDistance();
	bool updated = false;

	float currentDistance = distances[pointIndex]; // by default radius + FLT_MIN
	float tentativeDistance = distances[neighIndex] + distanceBetween;

	// If the tentative distance is shorter than the current distance and within the radius of the source,
	// update the current point with the tentative distance 
	if (tentativeDistance < currentDistance) {
		distances[pointIndex] = tentativeDistance;
		updated = true;
	}

	return updated;
}