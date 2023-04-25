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

__device__ __inline__ bool isNearBoundary(const bool* boundary, const int pixelIndex, int cols) {
	int x = 0, y = 0;
	indexToCoords(pixelIndex, x, y, cols);

	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			int newX = x + i;
			int newY = y + j;

			//bool xInBounds = newX < cols && newX >= 0;
			//bool yInBounds = newY < rows && newY >= 0;
			//bool inBounds = xInBounds && yInBounds;

			int neighIndex = coordsToIndex(newX, newY, cols);

			if (boundary[neighIndex])
				return true;
		}
	}

	return false;
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

		if (nextGoal == -1)
			return -1;

		distance += indexDistance(currentPoint, nextGoal, cols);
		currentPoint = nextGoal;
		nextGoal = distanceMap[currentPoint].second;
	}

	return distance;
}

__device__ __inline__ float stepDistance(CUDAPair<float, int>* distanceMap, int pointIndex, int cols) {
	float distance = 0.0f;

	int currentPoint = pointIndex;
	int nextGoal = distanceMap[currentPoint].second;

	while (nextGoal != currentPoint) {

		distance += distanceMap[currentPoint].first;

		currentPoint = nextGoal;
		nextGoal = distanceMap[currentPoint].second;
	}

	return distance;
}

#endif // CUDACOMMON_CUH