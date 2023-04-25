#pragma once

#ifndef EUCLIDEAN_EXPANSION_H
#define EUCLIDEAN_EXPANSION_H

#include "Common.h"
#include "EuclideanExpansionK.cuh"

void runEuclideanExpansion(config configuration);

float* computeCoverageMap(const bool* boundary, const int* sourceDistribution, float radius, int rows, int cols, int numSources);

#endif

