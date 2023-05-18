#pragma once

//#ifndef CUDACOMMON_CUH
//#define CUDACOMMON_CUH

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cfloat>
#include "MapElement.cuh"

constexpr float SCALE = 1.0;

__device__ __inline__ int getThreadId() {
	// Compute the block ID by multiplying the y coordinate of the block by the number of columns in the grid and adding the x coordinate.
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;

	// Compute the thread ID by multiplying the block ID by the total number of threads per block and adding the thread ID within the block.
	int threadId = blockId * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	// Return the computed thread ID.
	return threadId;
}

__device__ __inline__ void get2DThreadId(int& tidX, int& tidY) {
	tidX = blockIdx.x * blockDim.x + threadIdx.x;
	tidY = blockIdx.y * blockDim.y + threadIdx.y;
}

__device__ __inline__ int det(int a11, int a12, int a21, int a22) {
	return a11 * a22 - a12 * a21;
}

__device__ __inline__ int sgn(int x) {
	return (x > 0) - (x < 0);
}

__device__ __inline__ void indexToCoords(int index, int& x, int& y, int cols)
{
	// Compute the x coordinate as the remainder of the index divided by the number of columns.
	x = index % cols;

	// Compute the y coordinate as the integer division of the index by the number of columns.
	y = index / cols;
}

__device__ __inline__ int coordsToIndex(int x, int y, int cols)
{
	// Compute the flat index by multiplying the y coordinate by the number of columns and adding the x coordinate.
	return y * cols + x;
}

__global__ __inline__ void extractBoundary(const float* image, bool* boundary, int numElements) {
	// Determine pixel to be processed by thread.
	int tid = getThreadId();

	if (tid < numElements) {

		// Set the normalized value to true/false depending if it's a wall/interior point.
		if (image[tid] >= 127.5)
			boundary[tid] = false;
		else
			boundary[tid] = true;
	}
}

__device__ __inline__ bool isCorner(const float* rawImage, const int pixelIndex, int rows, int cols) {
	int x = 0, y = 0;
	indexToCoords(pixelIndex, x, y, cols);

	int hits = 0;
	int neigh = 0;

	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			int newX = x + i;
			int newY = y + j;

			bool xInBounds = newX < cols && newX >= 0;
			bool yInBounds = newY < rows && newY >= 0;
			bool inBounds = xInBounds && yInBounds;

			int neighIndex = coordsToIndex(newX, newY, cols);

			if (inBounds)
				if (rawImage[neighIndex] <= 127.5f) {
					neigh++;
					if (i * j != 0)
						hits++;
				}
		}
	}

	return hits == 1 && neigh > 1;
}

__global__ __inline__ void preProcessDomain(const float* rawImage, int* domain, int rows, int cols) {
	int tid = getThreadId();

	int numElements = rows * cols;

	if (tid < numElements) {
		float pixelIntensity = rawImage[tid];

		if (pixelIntensity <= 127.5f) // Wall
			domain[tid] = -1;
		else if (isCorner(rawImage, tid, rows, cols))
			domain[tid] = 1; // Interior point corner
		else
			domain[tid] = 0; // Interior point free
	}
}

__global__ __inline__ void processResultsRGB_(const int* boundary,  const MapElement* distanceMap, float* outputR, float* outputG, float* outputB, const float radius, int numElements, int cols) {
	// Determine pixel to be processed by thread.
	int tid = getThreadId();

	// Check that the value is within the boundaries.
	if (tid < numElements) {

		bool isWall = boundary[tid] == -1;
		float pixelValue = distanceMap[tid].distance;

		if (isWall ) {
			outputR[tid] = 0.0f;
			outputG[tid] = 0.0f;
			outputB[tid] = 0.0f;
		}
		else {
			// Set different colors based on the pixel value.
			if (pixelValue < (radius + FLT_MIN)) {
				float R, G, B;

				if (pixelValue < 4) {
					R = 255;
					G = 0;
					B = 0;
				}
				else {	
					float tint = (1.0 - pixelValue / radius);

					R = 149.0f;
					G = 152.0f;
					B = 144.0f;

					R = R + (255.0f - R) * tint;
					G = G + (255.0f - G) * tint;
					B = B + (255.0f - B) * tint;
				}

				outputR[tid] = R/255;
				outputG[tid] = G/255;
				outputB[tid] = B/255;
			}
			else {
				outputR[tid] = 255.0f/255;
				outputG[tid] = 255.0f/255;
				outputB[tid] = 255.0f/255;
			}
		}
	}
}

__global__ __inline__ void getBlockID(float* blockIDs, int rows, int cols) {
	// Determine pixel to be processed by thread.
	//int tid = getThreadId();

	int tidX, tidY;
	get2DThreadId(tidX, tidY);

	// Check that the value is within the boundaries.
	if (tidX < cols && tidY < rows) {
		int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	
		int tid = coordsToIndex(tidX, tidY, cols);
		blockIDs[tid] = static_cast<float>(blockId);
	}
}

__device__ __inline__ bool isCorner(const bool* boundary, const int pixelIndex, int rows, int cols) {
	int x = 0, y = 0;
	indexToCoords(pixelIndex, x, y, cols);
	
	int hits = 0;
	int neigh = 0;

	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			int newX = x + i;
			int newY = y + j;

			bool xInBounds = newX < cols && newX >= 0;
			bool yInBounds = newY < rows && newY >= 0;
			bool inBounds = xInBounds && yInBounds;

			int neighIndex = coordsToIndex(newX, newY, cols);

		//	if (i * j != 0)
				if (inBounds)
					if (boundary[neighIndex]) {
						neigh++;
						if (i * j != 0)
							hits++;
					}
		}
	}

	return hits == 1 && neigh > 1;
}

// DISTANCES
__device__ __inline__ float unitStepDistance() {
	// Return the distance corresponding to one unit length step, multiplied by the scaling factor.
	return SCALE * 1.0;
}

__device__ __inline__ float diagStepDistance() {
	// Return the distance corresponding to one diagonal step, multiplied by the scaling factor.
	return SCALE * sqrtf(2);
}

__device__ __inline__ float indexDistance(int pointA, int pointB, int cols) {
	int xA, xB, yA, yB;

	indexToCoords(pointA, xA, yA, cols);
	indexToCoords(pointB, xB, yB, cols);

	return sqrtf((xA - xB) * (xA - xB) + (yA - yB) * (yA - yB));
}

__device__ __inline__ float computeDistance(MapElement* distanceMap, int pointIndex, int cols) {
	float distance = 0.0f;

	int currentPoint = pointIndex;
	int nextGoal = distanceMap[currentPoint].predecessor;

	while (nextGoal != currentPoint) {
		distance += indexDistance(currentPoint, nextGoal, cols);
		currentPoint = nextGoal;
		nextGoal = distanceMap[currentPoint].predecessor;
	}

	return distance;
}


//#endif // CUDACOMMON_CUH