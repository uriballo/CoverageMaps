#pragma once

//#ifndef FEEDBACK_EXPANSION_H
//#define FEEDBACK_EXPANSION_H

#include "Common.h"
#include "FeedbackExpansionK.cuh"

void runExactExpansion(configuration& config);

MapElement* initialCoverageMapGPU(std::vector<int> servicesDistribution, int numServices, int rows, int cols, float initRadius, int initPredecessor);

int* getDomainGPU(const int* hostDomain, int numElements);

MapElement* computeCoverage(cudaTextureObject_t domainTexture, MapElement* deviceCoverageMap, configuration& config, int rows, int cols);

//#endif

