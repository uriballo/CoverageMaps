#include "EuclideanExpansion.h"

void runEuclideanExpansion(SystemParameters config) {
	/*
	int rows, cols;

	bool* boundary = IO::extractImageBoundary(config.imagePath, rows, cols);

	IO::writeBoolMatrix(boundary, rows, cols, "boundary");

	std::cout << "Domain dimensions: (" << cols << " x " << rows << ")" << std::endl;

	std::vector<int> servicesDistribution;

	if (config.customDistribution) {
		servicesDistribution = UTILS::convertString2IntVector(config.serviceDistribution);
	}
	else {
		// TODO: UPDATE to INT*
		//servicesDistribution = UTILS::getRandomSourceDistribution(boundary, rows, cols, config.numberOfServices);
	}

	float* coverageMap = computeCoverageMap(boundary, servicesDistribution, config.serviceRadius, rows, cols, config.numberOfServices);

	IO::writeFloatMatrix(coverageMap, rows, cols, "coverage-map");

	//TODO: UPDATE to RGB
//	float* processedResult = UTILS::processResults(boundary, coverageMap, config.serviceRadius, rows, cols);
//	cv::Mat map = IO::floatToCV(processedResult, rows, cols);
//	IO::storeBWImage(map, "Coverage Map");

//	delete[] processedResult;

	delete[] coverageMap;
	delete[] boundary;
	*/
}

float* computeCoverageMap(const bool* boundary, const std::vector<int>& sourceDistribution, float radius, int rows, int cols, int numSources) {
	bool hostFlag = false;
	bool* deviceFlag;

	const int numElements = rows * cols;

	// Initialize CoverageMap
	float* hostMap = new float[numElements];
	std::fill(hostMap, hostMap + numElements, radius + FLT_MIN);

	for (int i = 0; i < 2 * numSources; i += 2) {
		int index = sourceDistribution[i + 1] * cols + sourceDistribution[i];
		hostMap[index] = 0.0f;
	}

	// Device memory allocation
	float* deviceMap;
	CUDA::allocate(deviceMap, numElements);
	CUDA::copyHostToDevice(deviceMap, hostMap, numElements);

	bool* deviceBoundary;
	CUDA::allocate(deviceBoundary, numElements);
	CUDA::copyHostToDevice(deviceBoundary, boundary, numElements);

	CUDA::allocate(deviceFlag, 1);

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

	int iterations = 0;
	do {
		CUDA::set(deviceFlag, false);
		euclideanExpansionKernel << <blocksPerGrid, threadsPerBlock >> > (deviceBoundary, deviceMap, deviceFlag, rows, cols);
		CUDA::sync();
		CUDA::copyVarDeviceToHost(&hostFlag, deviceFlag);
	} while (hostFlag);

	CUDA::copyDeviceToHost(hostMap, deviceMap, numElements);

	CUDA::free(deviceMap);
	CUDA::free(deviceBoundary);

	return hostMap;
}


