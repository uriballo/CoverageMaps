#pragma once

#include "CUDACommon.cuh"

/**
 * @brief Performs a pseudoEuclidean expansion based on a domain to update a coverage map.
 *
 * The expansion is performed iteratively until no further changes occur.
 *
 * @param domainTex The CUDA texture object representing the domain data.
 * @param coverageMap A pointer to the coverage map where the results will be stored.
 * @param globalChanges A boolean flag indicating if any changes occurred during the expansion.
 * @param rows The number of rows in the domain texture.
 * @param cols The number of columns in the domain texture.
 */
__global__ void pseudoEuclideanExpansion(cudaTextureObject_t domainTex, MapElement * coverageMap,  bool* globalChanges, int rows, int cols);

/**
 * @brief Scans a 3x3 window centered at pixelIndex to see if it can improve its distance.
 *
 * This function scans a window of neighboring pixels around a given pixel in a domain texture.
 * The function returns a boolean indicating whether any distances were updated during the scan.
 *
 * @param domainTex The CUDA texture object representing the domain data.
 * @param coverageMap A pointer to the coverage map where the distances will be updated.
 * @param pixelIndex The index of the pixel in the domain texture being scanned.
 * @param rows The number of rows in the domain texture.
 * @param cols The number of columns in the domain texture.
 * @return A boolean indicating whether any distances were updated during the scan.
 */
__device__ bool scanWindow(cudaTextureObject_t domainTex, MapElement* coverageMap, const int pixelIndex, int rows, int cols);

/**
 * @brief Checks and if possible updates distances from a neighbour of a point.
 *
 * @param coverageMap A pointer to the coverage map where the distances will be updated.
 * @param pointIndex The index of the current point in the coverage map.
 * @param neighIndex The index of the neighboring point in the coverage map.
 * @param diag A boolean indicating whether the neighboring point is diagonal to the current point.
 * @return A boolean indicating whether any distances were updated.
 */
__device__ bool checkNeighInfo(MapElement* coverageMap, int pointIndex, int neighIndex, bool diag);
