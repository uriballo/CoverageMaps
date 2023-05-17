#pragma once

//#ifndef EUCLIDEAN_EXPANSION_H
//#define EUCLIDEAN_EXPANSION_H

#include "Common.h"
#include "EuclideanExpansionK.cuh"

void runEuclideanExpansion(configuration config);

float* computeCoverageMap(const bool* boundary, const std::vector<int>& sourceDistribution, float radius, int rows, int cols, int numSources);

//#endif

