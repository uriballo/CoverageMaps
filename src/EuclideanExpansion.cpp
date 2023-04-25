#include "EuclideanExpansion.h"

void runEuclideanExpansion(config conf) {
	int rows, cols;

	bool* boundary = IO::extractImageBoundary(conf.domainPath, rows, cols);

	IO::writeBoolMatrix(boundary, rows, cols, "boundary");

	std::cout << "Domain dimensions: (" << cols << " x " << rows << ")" << std::endl;

	int* sourceDistribution = conf.randomSources ?
		UTILS::getRandomSourceDistribution(boundary, rows, cols, conf.numSources)
		: conf.sources;

	float* coverageMap = computeCoverageMap(boundary, sourceDistribution, conf.radius, rows, cols, conf.numSources);

	IO::writeFloatMatrix(coverageMap, rows, cols, "coverage-map");

	if (conf.showResults) {
		if (conf.displayHeatMap)
			IO::showHeatMap(coverageMap, rows, cols);

		float* processedResult = UTILS::processResults(boundary, coverageMap, conf.radius, rows, cols);
		cv::Mat map = IO::floatToCV(processedResult, rows, cols);
		IO::showBW(map, "Coverage Map");

		delete[] processedResult;
	}

	delete[] coverageMap;
	delete[] boundary;
	delete[] sourceDistribution;
}

float* computeCoverageMap(const bool* boundary, const int* sourceDistribution, float radius, int rows, int cols, int numSources) {
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
		euclideanExpansionKernel << <blocksPerGrid, threadsPerBlock >> > (deviceBoundary, deviceMap, sourceDistribution, deviceFlag, rows, cols);
		CUDA::synchronize();
		CUDA::copyVarDeviceToHost(&hostFlag, deviceFlag);
	} while (hostFlag);

	CUDA::copyDeviceToHost(hostMap, deviceMap, numElements);

	CUDA::free(deviceMap);
	CUDA::free(deviceBoundary);

	return hostMap;
}


