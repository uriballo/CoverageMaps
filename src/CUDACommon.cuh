#pragma once

#ifndef CUDACOMMON_CUH
#define CUDACOMMON_CUH

#include <cuda_runtime.h>

#include <device_launch_parameters.h>
#include <cmath>
#include <cfloat>

#include "CUDAPair.cuh"

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

__global__ __inline__ void processResultsBW_(const bool* boundary, const CUDAPair<float, int>* distanceMap, float* output, const float radius, int numElements) {
	// Determine pixel to be processed by thread.
	int tid = getThreadId();

	// Check that the value is within the boundaries.
	if (tid < numElements) {

		bool isWall = boundary[tid];
		float pixelValue = distanceMap[tid].first;

		if (isWall) {
			output[tid] = 0.0f;
		}
		else {
			if (pixelValue != 0.0f)
				output[tid] = pixelValue < (radius + FLT_MIN) ? 1.0f : 0.5f;
		}
	}
}

/*
__device__ __inline__ void palette1(int id, float& R, float& G, float& B) {
	switch (id) {
	case 1:
		R = 96.0f;
		G = 108.0f;
		B = 56.0f;
		break;
	case 2:
		R = 40.0f;
		G = 54.0f;
		B = 24.0f;
		break;
	case 3:
		R = 254.0f;
		G = 250.0f;
		B = 224.0f;
		break;
	case 4:
		R = 221.0f;
		G = 161.0f;
		B = 94.0f;
		break;
	case 5:
		R = 188.0f;
		G = 108.0f;
		B = 37.0f;
		break;
	}
}

__device__ __inline__ int pointOrigin(const CUDAPair<float, int>* distanceMap, const int* sources, int pointIndex, int cols) {
	int currentPoint = pointIndex;
	int nextGoal = distanceMap[currentPoint].second;

	while (nextGoal != currentPoint) {
		currentPoint = nextGoal;
		nextGoal = distanceMap[currentPoint].second;
	}

	int j = 0, i = 0;
	int x, y;
	indexToCoords(currentPoint, x, y, cols);

	while (x != sources[i] && y != sources[i + 1]) {
		i += 2;
		j++;
	}

	return j;
}
*/

__global__ __inline__ void processResultsRGB_(const bool* boundary, const int* sources, const CUDAPair<float, int>* distanceMap, float* outputR, float* outputG, float* outputB, const float radius, int numElements, int cols) {
	// Determine pixel to be processed by thread.
	int tid = getThreadId();

	// Check that the value is within the boundaries.
	if (tid < numElements) {

		bool isWall = boundary[tid];
		float pixelValue = distanceMap[tid].first;

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
					// float tint = 1.0f / (1.0f + exp(-10.0f * (pixelValue / radius - 0.5f)));

					R = 149.0f;
					G = 152.0f;
					B = 144.0f;

					R = R + (255.0f - R) * tint;
					G = G + (255.0f - G) * tint;
					B = B + (255.0f - B) * tint;
				}
				
				//int id = 2;//pointOrigin(distanceMap, sources, tid, cols);
				//palette1(id, R, G, B);

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

__global__ __inline__ void processResultsBW(const bool* boundary, const float* distanceMap, float* output, const float radius, int numElements) {
	// Determine pixel to be processed by thread.
	int tid = getThreadId();

	// Check that the value is within the boundaries.
	if (tid < numElements) {

		bool isWall = boundary[tid];
		float pixelValue = distanceMap[tid];

		if (isWall) {
			output[tid] = 0.0f;
		}
		else {
			if (pixelValue != 0.0f)
				output[tid] = pixelValue < (radius + FLT_MIN) ? 1.0f : 0.5f;
		}
	}
}

__device__ __inline__ bool isNearBoundary(const bool* boundary, const int pixelIndex, int rows, int cols) {
	int x = 0, y = 0;
	indexToCoords(pixelIndex, x, y, cols);

	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			int newX = x + i;
			int newY = y + j;

			bool xInBounds = newX < cols && newX >= 0;
			bool yInBounds = newY < rows && newY >= 0;
			bool inBounds = xInBounds && yInBounds;

			int neighIndex = coordsToIndex(newX, newY, cols);
		//	if(newX * newY == 0)
				if(inBounds)
					if (boundary[neighIndex])
						return true;
		}
	}

	return false;
}

__device__ __inline__ bool isCorner(const bool* boundary, const int pixelIndex, int rows, int cols) {
	int x = 0, y = 0;
	indexToCoords(pixelIndex, x, y, cols);
	int hits = 0;

	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			int newX = x + i;
			int newY = y + j;

			bool xInBounds = newX < cols && newX >= 0;
			bool yInBounds = newY < rows && newY >= 0;
			bool inBounds = xInBounds && yInBounds;

			int neighIndex = coordsToIndex(newX, newY, cols);
			if (i * j != 0)
				if (inBounds)
					if (boundary[neighIndex])
						 hits++;
		}
	}
//	if (pixelIndex)
	return hits == 1;
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

__device__ __inline__ float computeDistance(CUDAPair<float, int>* distanceMap, int pointIndex, int cols) {
	float distance = 0.0f;

	int currentPoint = pointIndex;
	int nextGoal = distanceMap[currentPoint].second;

	while (nextGoal != currentPoint) {
		distance += indexDistance(currentPoint, nextGoal, cols);
		currentPoint = nextGoal;
		nextGoal = distanceMap[currentPoint].second;
	}

	return distance;
}

__device__ __inline__ bool visibilityTest(const bool* domain, int rows, int cols, int oX, int oY, int gX, int gY) {
	int dx = abs(gX - oX);
	int dy = abs(gY - oY);
	int sx = (oX < gX) ? 1 : -1;
	int sy = (oY < gY) ? 1 : -1;
	int err = dx - dy;

	while (true) {
		if (domain[coordsToIndex(oX, oY, cols)])
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

#endif // CUDACOMMON_CUH