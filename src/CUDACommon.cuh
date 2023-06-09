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

/*
  * @brief Get the ID of the current thread in the grid.
  * @return The computed thread ID.
*/
__device__ __inline__ int getThreadId() {
	// Compute the block ID by multiplying the y coordinate of the block by the number of columns in the grid and adding the x coordinate.
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;

	// Compute the thread ID by multiplying the block ID by the total number of threads per block and adding the thread ID within the block.
	int threadId = blockId * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	// Return the computed thread ID.
	return threadId;
}

/*
  * @brief Get the 2D thread ID in the grid.
  * @param tidX Reference to store the x coordinate of the thread.
  * @param tidY Reference to store the y coordinate of the thread.
*/
__device__ __inline__ void get2DThreadId(int& tidX, int& tidY) {
	tidX = blockIdx.x * blockDim.x + threadIdx.x;
	tidY = blockIdx.y * blockDim.y + threadIdx.y;
}

/*
  * @brief Compute the determinant of a 2x2 matrix.
  * @param a11 The element at row 1, column 1.
  * @param a12 The element at row 1, column 2.
  * @param a21 The element at row 2, column 1.
  * @param a22 The element at row 2, column 2.
  * @return The determinant of the matrix.
*/
__device__ __inline__ int det(int a11, int a12, int a21, int a22) {
	return a11 * a22 - a12 * a21;
}

/*
  * @brief Compute the determinant of a 2x2 matrix.
  * @param a11 The element at row 1, column 1.
  * @param a12 The element at row 1, column 2.
  * @param a21 The element at row 2, column 1.
  * @param a22 The element at row 2, column 2.
  * @return The determinant of the matrix.
*/
__device__ __inline__ int sgn(int x) {
	return (x > 0) - (x < 0);
}

/*
  * @brief Convert a flat index to 2D coordinates.
  * @param index The flat index to convert.
  * @param x Reference to store the computed x coordinate.
  * @param y Reference to store the computed y coordinate.
  * @param cols The number of columns in the grid.
*/
__device__ __inline__ void indexToCoords(int index, int& x, int& y, int cols)
{
	// Compute the x coordinate as the remainder of the index divided by the number of columns.
	x = index % cols;

	// Compute the y coordinate as the integer division of the index by the number of columns.
	y = index / cols;
}

/*
  * @brief Convert 2D coordinates to a flat index.
  * @param x The x coordinate.
  * @param y The y coordinate.
  * @param cols The number of columns in the grid.
  * @return The flat index corresponding to the coordinates.
*/
__device__ __inline__ int coordsToIndex(int x, int y, int cols)
{
	// Compute the flat index by multiplying the y coordinate by the number of columns and adding the x coordinate.
	return y * cols + x;
}

/*
  * @brief Check if a pixel is a corner based on its neighboring pixels.
  * @param grayscaleImage The grayscale image.
  * @param pixelIndex The index of the pixel.
  * @param rows The number of rows in the image.
  * @param cols The number of columns in the image.
  * @return True if the pixel is a corner, false otherwise.
*/
__device__ __inline__ bool checkIfCorner(const float* grayscaleImage, const int pixelIndex, int rows, int cols) {
	int x = 0, y = 0;
	indexToCoords(pixelIndex, x, y, cols); // Convert pixel index to coordinates

	int diagWallHits = 0; // Counter for corner hits
	int wallHits = 0; // Counter for neighboring pixels

	for (int i = -1; i < 2; i++) { // Loop over neighboring rows
		for (int j = -1; j < 2; j++) { // Loop over neighboring columns
			int newX = x + i; // Compute new x-coordinate
			int newY = y + j; // Compute new y-coordinate

			// Check if both x and y coordinates are within the image bounds
			bool xInBounds = newX < cols && newX >= 0;
			bool yInBounds = newY < rows && newY >= 0;
			bool inBounds = xInBounds && yInBounds;

			int neighIndex = coordsToIndex(newX, newY, cols);

			if (inBounds)
				if (grayscaleImage[neighIndex] <= 50.0f) { // Check if the intensity of the neighboring pixel is less than or equal to 127.5 (Wall)
					wallHits++;

					if (i * j != 0)
						diagWallHits++;
				}
		}
	}

	return diagWallHits == 1 && wallHits > 1;
}

/**
 * @brief Performs cell decomposition based on a grayscale image.
 *
 * Cells with pixel intensity <= 50.0 are considered occupied and assigned a domain value of -1.
 * Cells that are determined to be corners are assigned a domain value of 1.
 * All other cells are considered free points are assigned a domain value of 0.
 *
 * @param grayscaleImage A pointer to the grayscale image data.
 * @param domain A pointer to the domain array where the results will be stored.
 * @param rows The number of rows in the image.
 * @param cols The number of columns in the image.
 */
__global__ __inline__ void imageCellDecomposition(const float* grayscaleImage, int* domain, int rows, int cols) {
	int threadId = getThreadId();
	int numElements = rows * cols;

	if (threadId < numElements) {
		// Get the intensity of the pixel.
		float pixelIntensity = grayscaleImage[threadId];

		if (pixelIntensity <= 50.0f)
			// Assign -1 to the domain to represent a wall or obstacle.
			domain[threadId] = -1;
		else if (checkIfCorner(grayscaleImage, threadId, rows, cols))
			// Assign 1 to the domain to represent a free corner.
			domain[threadId] = 1;
		else
			// Assign 0 to the domain to represent a free point.
			domain[threadId] = 0;
	}
}

/*
  * @brief Pick a color based on the source index.
  * @param sourceIndex The index of the source.
  * @param sourceDistribution The distribution of sources.
  * @param numSources The number of sources.
  * @param R Reference to store the red component of the color.
  * @param G Reference to store the green component of the color.
  * @param B Reference to store the blue component of the color.
  * @param cols The number of columns in the image.
*/
__device__ __inline__ void pickColor(int sourceIndex, int* sourceDistribution, int numSources, float& R, float& G, float& B, int cols) {
	int id;
	int j = -1;
	for (int i = 0; i < numSources * 2; i += 2) {
		j++;

		id = coordsToIndex(sourceDistribution[i], sourceDistribution[i + 1], cols);

		if (id == sourceIndex)
			break;
	}

	int normalizedId = j % 10;

	switch (normalizedId) {
	case 0:
		R = 26;
		G = 64;
		B = 121;
		break;
	case 01:
		R = 203;
		G = 223;
		B = 144;
		break;
	case 2:
		R = 143;
		G = 173;
		B = 136;
		break;
	case 3:
		R = 77;
		G = 124;
		B = 138;
		break;
	case 4:
		R = 232;
		G = 232;
		B = 228;
		break;
	case 5:
		R = 216;
		G = 226;
		B = 220;
		break;
	case 6:
		R = 236;
		G = 228;
		B = 219;
		break;
	case 7:
		R = 255;
		G = 229;
		B = 217;
		break;
	case 8:
		R = 255;
		G = 215;
		B = 186;
		break;
	case 9:
		R = 254;
		G = 200;
		B = 154;
		break;
	}
}

/*
  * @brief Pick a color based on the source index.
  * @param sourceIndex The index of the source.
  * @param sourceDistribution The distribution of sources.
  * @param numSources The number of sources.
  * @param R Reference to store the red component of the color.
  * @param G Reference to store the green component of the color.
  * @param B Reference to store the blue component of the color.
  * @param cols The number of columns in the image.
*/
__device__ __inline__ void pickColor2(int sourceIndex, int* sourceDistribution, int numSources, float& R, float& G, float& B, int cols) {
	int id;
	int j = -1;
	for (int i = 0; i < numSources * 2; i += 2) {
		j++;

		id = coordsToIndex(sourceDistribution[i], sourceDistribution[i + 1], cols);

		if (id == sourceIndex)
			break;
	}

	int normalizedId = j % 10;
	switch (normalizedId) {
	case 0:
		R = 0;
		G = 124;
		B = 190;
		break;
	case 1:
		R = 6;
		G = 39;
		B = 38;
		break;
	case 2:
		R = 154;
		G = 125;
		B = 110;
		break;
	case 3:
		R = 176;
		G = 160;
		B = 136;
		break;
	case 4:
		R = 198;
		G = 195;
		B = 163;
		break;
	case 5:
		R = 85;
		G = 196;
		B = 49;
		break;
	case 6:
		R = 225;
		G = 234;
		B = 49;
		break;
	case 7:
		R = 128;
		G = 229;
		B = 205;
		break;
	case 8:
		R = 183;
		G = 141;
		B = 23;
		break;
	case 9:
		R = 219;
		G = 198;
		B = 139;
		break;
	}


}

/*
  * @brief Pick a color based on the source index.
  * @param sourceIndex The index of the source.
  * @param sourceDistribution The distribution of sources.
  * @param numSources The number of sources.
  * @param R Reference to store the red component of the color.
  * @param G Reference to store the green component of the color.
  * @param B Reference to store the blue component of the color.
  * @param cols The number of columns in the image.
*/
__global__ __inline__ void processResultsRGB_(const int* boundary, const MapElement* distanceMap, int* sourceDistribution, int numSources, float* outputR, float* outputG, float* outputB, const float radius, int numElements, int cols) {
	// Determine pixel to be processed by thread.
	int tid = getThreadId();

	// Check that the value is within the boundaries.
	if (tid < numElements) {

		bool isWall = boundary[tid] == -1;
		float pixelValue = distanceMap[tid].distance;

		if (isWall) {
			outputR[tid] = 0.0f;
			outputG[tid] = 0.0f;
			outputB[tid] = 0.0f;
		}
		else {
			// Set different colors based on the pixel value.
			if (pixelValue < (radius + FLT_MIN)) {
				float R, G, B;

				if (pixelValue < 3) {
					R = 255;
					G = 0;
					B = 0;
				}
				else {
					float tint = (1.0 - pixelValue / radius);

					pickColor2(distanceMap[tid].source, sourceDistribution, numSources, R, G, B, cols);


					R = R + (255.0f - R) * tint;
					G = G + (255.0f - G) * tint;
					B = B + (255.0f - B) * tint;
				}

				outputR[tid] = R / 255;
				outputG[tid] = G / 255;
				outputB[tid] = B / 255;
			}
			else {
				outputR[tid] = 255.0f / 255;
				outputG[tid] = 255.0f / 255;
				outputB[tid] = 255.0f / 255;
			}
		}
	}
}

// DISTANCES

/*
  * @brief Returns the distance corresponding to one unit length step, multiplied by the scaling factor.
  * @return The unit step distance.
*/
__device__ __inline__ float unitStepDistance() {
	// Return the distance corresponding to one unit length step, multiplied by the scaling factor.
	return SCALE * 1.0;
}

/*
  * @brief Returns the distance corresponding to one diagonal step, multiplied by the scaling factor.
  * @return The diagonal step distance.
*/
__device__ __inline__ float diagStepDistance() {
	// Return the distance corresponding to one diagonal step, multiplied by the scaling factor.
	return SCALE * sqrtf(2);
}

/*
  * @brief Computes the distance between two points based on their indices and the number of columns.
  * @param pointA The index of the first point.
  * @param pointB The index of the second point.
  * @param cols The number of columns in the image.
  * @return The distance between the two points.
*/
__device__ __inline__ float indexDistance(int pointA, int pointB, int cols) {
	int xA, xB, yA, yB;

	indexToCoords(pointA, xA, yA, cols);
	indexToCoords(pointB, xB, yB, cols);

	return sqrtf((xA - xB) * (xA - xB) + (yA - yB) * (yA - yB));
}

/*
  * @brief Computes the total distance along the path from a given point to its predecessor in the distance map.
  * @param distanceMap The distance map.
  * @param pointIndex The index of the current point.
  * @param cols The number of columns in the image.
  * @return The computed distance.
*/
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