#pragma once

//#ifndef FEEDBACK_EXPANSION_H
//#define FEEDBACK_EXPANSION_H

#include "Common.h"
#include "FeedbackExpansionK.cuh"

void runExactExpansion(configuration& config);

MapElement* initialCoverageMapGPU(std::vector<int> servicesDistribution, int numServices, int rows, int cols, float initRadius, int initPredecessor);

int* convertDomain2Texture(const int* hostDomain, int numElements);

cudaTextureObject_t convertDomain2Texture(const int* hostDomain, int rows, int cols, cudaArray** domainArray);

float coveragePercent2(cudaTextureObject_t domainTexture, int* serviceDistribution, int numServices, float radius, MapElement* deviceCoverageMap, int rows, int cols);

MapElement* computeCoverage(cudaTextureObject_t domainTexture, MapElement* deviceCoverageMap, configuration& config, int rows, int cols);

MapElement* initialCoverageMapGPU(std::vector<int> servicesDistribution, MapElement* deviceCoverageMap, int numServices, int rows, int cols, float initRadius, int initPredecessor);

float computeCoveragePercent(cudaTextureObject_t domainTexture, MapElement* deviceCoverageMap, int rows, int cols, float radius);
//#endif

