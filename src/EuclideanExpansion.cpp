#include "EuclideanExpansion.h"

std::vector<int> getServiceDistribution(const SystemParameters& config, const int* domain, int numServices, int rows, int cols) {
	if (config.customDistribution) {
		return UTILS::convertString2IntVector(config.serviceDistribution);
	}
	else {
		return UTILS::generateRandomDistribution(domain, rows, cols, numServices);
	}
}

double parallelBellmanFord1(SystemParameters& config, cudaTextureObject_t domainTexture, MapElement* deviceCoverageMap, int rows, int cols, float radius) {
	auto startTime = std::chrono::steady_clock::now();

	bool hostFlag = false;
	bool* deviceFlag;

	int numElements = rows * cols;

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

	config.solutionData += "Number of threads per block: " + std::to_string(threadsPerBlock.x * threadsPerBlock.y) + "\n";
	config.solutionData += "Number of blocks: " + std::to_string(blocksPerGrid.x * blocksPerGrid.y) + "\n";

	int iterations = 1;

	CUDA::allocate(deviceFlag, 1);

	MapElement* intermediateResult = new MapElement[numElements];

	bool storeData = config.storeIterCoverage;

	do {
		CUDA::set(deviceFlag, false);
		pseudoEuclideanExpansion << <blocksPerGrid, threadsPerBlock >> > (domainTexture, deviceCoverageMap, deviceFlag, rows, cols);
		CUDA::sync();
		CUDA::copyDeviceToHost(&hostFlag, deviceFlag, 1);

		if (storeData) {
			CUDA::copyDeviceToHost(intermediateResult, deviceCoverageMap, numElements);
			std::string fileName = "iteration_" + std::to_string(iterations);
			IO::writeCoverageMap(intermediateResult, rows, cols, fileName);
		}

		iterations++;
	} while (hostFlag);

	config.solutionData += "Total iterations: " + std::to_string(iterations) + "\n\n";

	auto endTime = std::chrono::steady_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
	double seconds = duration.count() / 1000.0;

	CUDA::free(deviceFlag);

	return seconds;
}

void runEuclideanExpansion(SystemParameters& config, const AlgorithmParameters& algParams, const OptimizationParameters& optParams) {
	int rows, cols;
	config.solutionData = "";

	auto startTime = std::chrono::steady_clock::now();

	int* domain = COVERAGE::getDomainFromImage(config.imagePath, rows, cols);

	auto endTime = std::chrono::steady_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
	double seconds = duration.count() / 1000.0;

	int numElements = rows * cols;

	config.solutionData += "Domain process time: " + std::to_string(seconds) + " (s)\n";
	config.solutionData += "Domain dimensions: " + std::to_string(rows) + " x " + std::to_string(cols) + " (" + std::to_string(numElements) + " pixels) \n\n";

	MapElement* deviceCoverageMap = COVERAGE::getEmptyCoverageMap(numElements);

	std::vector<int> servicesDistribution = getServiceDistribution(config, domain, algParams.numServices, rows, cols);
	COVERAGE::setupCoverageMap(deviceCoverageMap, servicesDistribution, algParams.serviceRadius, rows, cols);

	cudaArray* domainArray;
	cudaTextureObject_t domainTexture = COVERAGE::convertDomainToTexture(domain, &domainArray, rows, cols);

	seconds = parallelBellmanFord1(config, domainTexture, deviceCoverageMap, rows, cols, algParams.serviceRadius);

	float coverage = COVERAGE::getCoveragePercent(domainTexture, deviceCoverageMap, rows, cols);
	config.solutionData += "Coverage map compute time: " + std::to_string(seconds) + " (s)\n";
	config.solutionData += "Coverage percent: " + std::to_string(coverage) + " %\n";
	config.solutionData += "Services distribution: " + UTILS::convertIntVector2String(servicesDistribution) + " \n";


	cv::Mat processedResultsRGB = UTILS::processResultsRGB(domain, deviceCoverageMap, servicesDistribution.data(), algParams.numServices, algParams.serviceRadius, rows, cols);

	IO::writeRGBImage(processedResultsRGB, "output/" + config.imageName);

	if (config.storeBoundary)
		IO::writeIntMatrix(domain, rows, cols, "domain");

	UTILS::freeExpansionResources(domainTexture, domainArray, deviceCoverageMap, domain);
}

