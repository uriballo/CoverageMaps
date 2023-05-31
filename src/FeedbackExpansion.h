#pragma once

//#ifndef FEEDBACK_EXPANSION_H
//#define FEEDBACK_EXPANSION_H

#include "MCLPSolver.h"

void runExactExpansion(SystemParameters& config, const AlgorithmParameters& algParams, const OptimizationParameters& optParams);

double computeCoverageMap(SystemParameters& config, cudaTextureObject_t domainTexture, MapElement* deviceCoverageMap, int rows, int cols, float radius);

//#endif

