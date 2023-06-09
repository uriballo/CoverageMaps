#pragma once

//#ifndef FEEDBACK_EXP_K_CUH
//#define FEEDBACK_EXP_K_CUH

#include "CUDACommon.cuh"
#include <iostream>
#include <cuda_texture_types.h>

/**
 * @brief Perform Euclidean expansion on the coverage map using CUDA.
 *
 * @param domainTex The CUDA texture object representing the input domain.
 * @param coverageMap The coverage map to be expanded.
 * @param globalChanges A boolean flag indicating if any changes were made to the coverage map.
 * @param rows The number of rows in the coverage map.
 * @param cols The number of columns in the coverage map.
 * @param radius The expansion radius.
 */
__global__ void euclideanExpansion(cudaTextureObject_t domainTex, MapElement* coverageMap, bool* globalChanges, int rows, int cols, float radius);

/**
 * @brief Scan the neighboring cells within a window and update the coverage map.
 *
 * @param domainTex The CUDA texture object representing the input domain.
 * @param coverageMap The coverage map to be updated.
 * @param pointIndex The index of the current cell.
 * @param rows The number of rows in the coverage map.
 * @param cols The number of columns in the coverage map.
 * @param radius The expansion radius.
 * @return True if any updates were made to the coverage map, false otherwise.
 */
__device__ bool scanWindow(cudaTextureObject_t domainTex, MapElement* coverageMap, int pointIndex, int rows, int cols, float radius);

/**
 * @brief Check the neighbor information and update the current cell's coverage information if necessary.
 *
 * @param domainTex The CUDA texture object representing the input domain.
 * @param pointInfo The coverage information of the current cell.
 * @param neighInfo The coverage information of the neighboring cell.
 * @param pointIndex The index of the current cell.
 * @param neighIndex The index of the neighboring cell.
 * @param rows The number of rows in the coverage map.
 * @param cols The number of columns in the coverage map.
 * @param radius The expansion radius.
 * @return True if the coverage information was updated, false otherwise.
 */
__device__ bool checkNeighInfo(cudaTextureObject_t domainTex, MapElement& pointInfo, MapElement neighInfo, int pointIndex, int neighIndex, int rows, int cols, float radius);

/**
 * @brief Check if the coverage information can be updated based on the neighbor information.
 *
 * @param domainTex The CUDA texture object representing the input domain.
 * @param coverageMap The coverage map.
 * @param pointIndex The index of the current cell.
 * @param neighIndex The index of the neighboring cell.
 * @param firstPredecessor The index of the first predecessor cell.
 * @param secondPredecessor The index of the second predecessor cell.
 * @param rows The number of rows in the coverage map.
 * @param cols The number of columns in the coverage map.
 * @return True if the coverage information can be updated, false otherwise.
 */
__device__ bool canUpdateInfo(cudaTextureObject_t domainTex, MapElement* coverageMap, int point, int neigh, int firstPredecessor, int secondPredecessor, int rows, int cols);

/**
 * @brief Determine a suitable predecessor cell for updating the coverage information.
 *
 * @param domainTex The CUDA texture object representing the input domain.
 * @param pointIndex The index of the current cell.
 * @param neighIndex The index of the neighboring cell.
 * @param neighPredecessorIndex The index of the neighboring cell's predecessor.
 * @param rows The number of rows in the coverage map.
 * @param cols The number of columns in the coverage map.
 * @return The index of the suitable predecessor cell.
 */
__device__ int suitablePredecessor(cudaTextureObject_t domainTex, int pointIndex,int neighIndex, int neighPredecessorIndex, int rows, int cols);

/**
 * @brief Perform the Exact Euclidean Distance Transform (EEDT) on the coverage map using CUDA.
 *
 * @param coverageMap The coverage map to be processed.
 * @param globalChanges A boolean flag indicating if any changes were made to the coverage map.
 * @param rows The number of rows in the coverage map.
 * @param cols The number of columns in the coverage map.
 * @param radius The expansion radius.
 */
__global__ void EEDT(MapElement* coverageMap, bool* globalChanges, int rows, int cols, float radius);

/**
 * @brief Initialize the coverage map with initial values.
 *
 * @param coverageMap The coverage map to be initialized.
 * @param initRadius The initial radius value.
 * @param initPredecessor The initial predecessor value.
 * @param servicesDistribution The distribution of services.
 * @param numServices The number of services.
 * @param numElements The number of elements in the coverage map.
 * @param cols The number of columns in the coverage map.
 */
__global__ void initializeCoverageMap(MapElement* coverageMap, float initRadius, int initPredecessor, int* servicesDistribution, int numServices, int numElements, int cols);


/**
 * @brief Evaluate the coverage of the input domain using the coverage map.
 *
 * @param domainTex The CUDA texture object representing the input domain.
 * @param coverageMap The coverage map.
 * @param rows The number of rows in the coverage map.
 * @param cols The number of columns in the coverage map.
 * @param interiorPoints The number of interior points.
 * @param coveredPoints The number of covered points.
 */
__global__ void evalCoverage(cudaTextureObject_t domainTex, MapElement* coverageMap, int rows, int cols, int* interiorPoints, int* coveredPoints);


/**
 * @brief Evaluate the coverage of the input domain using the coverage map.
 *
 * @param domainTex The CUDA texture object representing the input domain.
 * @param coverageMap The coverage map.
 * @param rows The number of rows in the coverage map.
 * @param cols The number of columns in the coverage map.
 * @param interiorPoints The number of interior points.
 * @param coveredPoints The number of covered points.
 */
__device__ bool visibilityTest(cudaTextureObject_t domainTex, int rows, int cols, int oX, int oY, int gX, int gY);


//#endif
