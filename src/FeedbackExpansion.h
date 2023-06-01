#pragma once

#include "Coverage.h"

double parallelBellmanFord2(SystemParameters& config, cudaTextureObject_t domainTexture, MapElement* deviceCoverageMap, int rows, int cols, float radius);

void runExactExpansion(SystemParameters& config, const AlgorithmParameters& algParams, const OptimizationParameters& optParams);

