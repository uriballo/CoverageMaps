#pragma once

#include "Coverage.h"

/*
  * @brief Runs a parallel version of the Bellman-Ford algorithm.
  *
  * This function implements the parallel Bellman-Ford algorithm via exact-Euclidean expansion.
  * 
  * @param config The system parameters.
  * @param domainTexture The domain texture object.
  * @param deviceCoverageMap The device coverage map.
  * @param rows The number of rows in the coverage map.
  * @param cols The number of columns in the coverage map.
  * @param radius The radius for processing.
  * @return The execution time in seconds.
*/
double parallelBellmanFord2(SystemParameters& config, cudaTextureObject_t domainTexture, MapElement* deviceCoverageMap, int rows, int cols, float radius);

/*
  * @brief Runs the exact expansion algorithm.
  * @param config The system parameters.
  * @param algParams The algorithm parameters.
  * @param optParams The optimization parameters.
*/
void runExactExpansion(SystemParameters& config, const AlgorithmParameters& algParams, const OptimizationParameters& optParams);

