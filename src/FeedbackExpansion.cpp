#include "FeedbackExpansion.h"

void runExactExpansion(config conf) {
	int rows, cols;

	bool* boundary = IO::extractImageBoundary(conf.domainPath, rows, cols);

	IO::writeBoolMatrix(boundary, rows, cols, "boundary");

	std::cout << "Domain dimensions: (" << cols << " x " << rows << ")" << std::endl;

	int* sourceDistribution = conf.randomSources ?
		UTILS::getRandomSourceDistribution(boundary, rows, cols, conf.numSources)
		: conf.sources;

	CUDAPair<float, int>* coverageMap = computeCoverage(boundary, sourceDistribution, conf.radius, rows, cols, conf.numSources);

	if (conf.showResults) {
		float* processedResult = UTILS::processResults_(boundary, coverageMap, conf.radius, rows, cols);
		cv::Mat map = IO::floatToCV(processedResult, rows, cols);
		IO::showBW(map, "Coverage Map");

		IO::writeFloatMatrix(processedResult, rows, cols, "processed-results");
		delete[] processedResult;
	}

	delete[] coverageMap;
	delete[] boundary;
	delete[] sourceDistribution;
}

CUDAPair<float, int>* computeCoverage(const bool* boundary, const int* sourceDistribution, float radius, int rows, int cols, int numSources) {
	bool hostFlag = false;

	bool* deviceFlag;

	const int numElements = rows * cols;

	CUDAPair<float, int>* hostCoverageMap = new CUDAPair<float, int>[numElements];

	// TODO: Merge
	UTILS::initializeCoverageMap(hostCoverageMap, radius + FLT_MIN, -1, numElements);
	UTILS::initializeSources(hostCoverageMap, sourceDistribution, numSources, cols);

	CUDAPair<float, int>* deviceCoverageMap;
	CUDA::allocateAndCopy(deviceCoverageMap, hostCoverageMap, numElements);

	bool* deviceBoundary;
	CUDA::allocateAndCopy(deviceBoundary, boundary, numElements);

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

	int iterations = 0;
	CUDA::allocate(deviceFlag, 1);

	//do {
	do {

		CUDA::set(deviceFlag, false);
		euclideanExpansion << <blocksPerGrid, threadsPerBlock >> > (deviceBoundary, deviceCoverageMap, deviceFlag, rows, cols, radius);
		CUDA::synchronize();
		CUDA::copyDeviceToHost(&hostFlag, deviceFlag, 1);
	} while (hostFlag);

	iterations++;
	//	std::cout << "hellno\n";
	//	CUDA::set(deviceFlag, false);
	//	EEDT << <blocksPerGrid, threadsPerBlock >> > (deviceBoundary, deviceCoverageMap, deviceFlag, rows, cols, radius);
	//	CUDA::synchronize();
	//	CUDA::copyDeviceToHost(&hostFlag, deviceFlag, 1);
	//} while (hostFlag);

	std::cout << std::endl << "Iterations: " << iterations << std::endl;

	CUDA::copyDeviceToHost(hostCoverageMap, deviceCoverageMap, numElements);
	CUDA::free(deviceBoundary);
	CUDA::free(deviceCoverageMap);
	CUDA::free(deviceFlag);

	return hostCoverageMap;
}
