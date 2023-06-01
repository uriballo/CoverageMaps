#pragma once

#include "Coverage.h"

/**
 * @brief Runs a parallel version of the Bellman-Ford algorithm.
 *
 * This function implements the parallel Bellman-Ford algorithm via pseudo-Euclidean expansion.
 *
 * @param config System parameters and solution data.
 * @param domainTexture The CUDA texture object representing the domain data.
 * @param deviceCoverageMap A pointer to the coverage map on the device.
 * @param rows The number of rows in the domain texture.
 * @param cols The number of columns in the domain texture.
 * @param radius The radius parameter for the expansion.
 * @return The elapsed time in seconds for the expansion process.
 */
double parallelBellmanFord1(SystemParameters& config, cudaTextureObject_t domainTexture, MapElement* deviceCoverageMap, int rows, int cols, float radius);

void runEuclideanExpansion(SystemParameters& config, const AlgorithmParameters& algParams, const OptimizationParameters& optParams);
