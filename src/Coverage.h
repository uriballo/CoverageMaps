#pragma once

#include "Common.h"
#include "FeedbackExpansionK.cuh"
#include "EuclideanExpansionK.cuh"
#include "MCLPSolver.h"

namespace COVERAGE {

	/*
	  * @brief Reads an image file, converts it to a domain representation, and returns the domain as an integer array.
	  * @param filePath The path to the image file.
	  * @param rows [out] The number of rows in the image.
	  * @param cols [out] The number of columns in the image.
	  * @return The domain representation of the image as an integer array.
	*/
	int* getDomainFromImage(std::string filePath, int& rows, int& cols);

	/*
	  * @brief Converts a domain represented by an integer array into a CUDA texture object.
	  * @param domain The domain represented as an integer array.
	  * @param domainArray [out] The CUDA array representation of the domain.
	  * @param rows The number of rows in the domain.
	  * @param cols The number of columns in the domain.
	  * @return The CUDA texture object representing the domain.
	*/
	cudaTextureObject_t convertDomainToTexture(int* domain, cudaArray** domainArray, int rows, int cols);

	/*
	  * @brief Creates an empty coverage map on the device with the specified number of elements.
	  * @param numElements The number of elements in the coverage map.
	  * @return The pointer to the empty coverage map on the device.
	*/
	MapElement* getEmptyCoverageMap(int numElements);

	/*
	  * @brief Sets up the initial state of the coverage map with the given services distribution, radius, and dimensions.
	  * @param deviceCoverageMap The pointer to the coverage map on the device.
	  * @param servicesDistribution The distribution of services as a vector of integers.
	  * @param radius The coverage radius.
	  * @param rows The number of rows in the coverage map.
	  * @param cols The number of columns in the coverage map.
	*/
	void setupCoverageMap(MapElement* coverageMap, std::vector<int> servicesDistribution, float radius, int rows, int cols);

	/*
	  * @brief Computes the exact coverage map by expanding the coverage until convergence.
	  * @param domain The CUDA texture object representing the domain.
	  * @param emptyCoverageMap The pointer to the empty coverage map on the device.
	  * @param radius The coverage radius.
	  * @param rows The number of rows in the coverage map.
	  * @param cols The number of columns in the coverage map.
	*/
	void computeExactCoverageMap(cudaTextureObject_t domain, MapElement* emptyCoverageMap, float radius, int rows, int cols);

	/*
	  * @brief Computes the coverage percentage of the given domain using the coverage map.
	  * @param domain The CUDA texture object representing the domain.
	  * @param coverageMap The pointer to the coverage map on the device.
	  * @param rows The number of rows in the coverage map.
	  * @param cols The number of columns in the coverage map.
	  * @return The coverage percentage.
	*/
	float getCoveragePercent(cudaTextureObject_t domain, MapElement* coverageMap, int rows, int cols);

}