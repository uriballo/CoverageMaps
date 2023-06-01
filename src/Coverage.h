#pragma once

#include "Common.h"
#include "FeedbackExpansionK.cuh"
#include "EuclideanExpansionK.cuh"
#include "MCLPSolver.h"

namespace COVERAGE {
	// INPUTS

	int* getDomainFromImage(std::string filePath, int& rows, int& cols);
	
	cudaTextureObject_t convertDomainToTexture(int* domain, cudaArray** domainArray, int rows, int cols);

	MapElement* getEmptyCoverageMap(int numElements);

	// MANAGEMENT
	void setupCoverageMap(MapElement* coverageMap, std::vector<int> servicesDistribution, float radius, int rows, int cols);
	
	// OPERATIONS
	void computeExactCoverageMap(cudaTextureObject_t domain, MapElement* emptyCoverageMap, float radius, int rows, int cols);

	float getCoveragePercent(cudaTextureObject_t domain, MapElement* coverageMap, int rows, int cols);

}