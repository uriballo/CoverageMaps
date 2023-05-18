#include "FeedbackExpansion.h"

void runExactExpansion(configuration& config) {
	int rows, cols;

	int* domain = IO::preProcessDomainImage(config.imagePath, rows, cols);

	int numElements = rows * cols;
	
	if (config.storeBoundary) {
		IO::writeIntMatrix(domain, rows, cols, "domain");
	}

	std::vector<int> servicesDistribution;

	if (config.customDistribution) {
		servicesDistribution = UTILS::convertStringToIntVector(config.serviceDistribution);
	}
	else {
		servicesDistribution = UTILS::getRandomSourceDistribution(domain, rows, cols, config.numberOfServices);
	}

	config.solutionData = "";
	config.solutionData += "Domain dimensions: " + std::to_string(rows) + " x " + std::to_string(cols) + " ("+ std::to_string(numElements) + " pixels) \n\n";

	int* deviceDomain = getDomainGPU(domain, numElements);
	MapElement* deviceCoverageMap = initialCoverageMapGPU(servicesDistribution, config.numberOfServices, rows, cols, config.serviceRadius + FLT_MIN, -1);

	auto startTime = std::chrono::steady_clock::now();

	MapElement* coverageMap = computeCoverage(deviceDomain, deviceCoverageMap, config, rows, cols);// computeCoverage(domain, servicesDistribution, config, rows, cols);

	auto endTime = std::chrono::steady_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
	double seconds = duration.count() / 1000.0;

	config.solutionData += "Coverage map compute time: " + std::to_string(seconds) + " (s)\n";

	cv::Mat processedResultsRGB = UTILS::processResultsRGB(domain, coverageMap, config.serviceRadius, rows, cols, config.numberOfServices);
	IO::storeRGB(processedResultsRGB, "output/" + config.imageName);

	delete[] coverageMap;
	delete[] domain;
}

MapElement* initialCoverageMapGPU(std::vector<int> servicesDistribution, int numServices, int rows, int cols, float initRadius, int initPredecessor) {
	int numElements = rows * cols;

	int* servicesArray = servicesDistribution.data();

	int* deviceServices;
	CUDA::allocateAndCopy(deviceServices, servicesArray, numServices * 2);

	MapElement* deviceCoverageMap;
	CUDA::allocate(deviceCoverageMap, numElements);

	dim3 threadsPerBlock(32, 32);
	dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

	initCoverageMap << <blocksPerGrid, threadsPerBlock >> > (deviceCoverageMap, initRadius, initPredecessor, deviceServices, numServices, numElements, cols);
	CUDA::synchronize();

	return deviceCoverageMap;
}

int* getDomainGPU(const int* hostDomain, int numElements) {
	int* deviceDomain;
	CUDA::allocateAndCopy(deviceDomain, hostDomain, numElements);

	return deviceDomain;
}

MapElement* computeCoverage(int* deviceBoundary, MapElement* deviceCoverageMap, configuration& config, int rows, int cols) {
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
		int innerIterations = 1;

		do {
			CUDA::set(deviceFlag, false);
			euclideanExpansion << <blocksPerGrid, threadsPerBlock >> > (deviceBoundary, deviceCoverageMap, deviceFlag, rows, cols, config.serviceRadius);
			CUDA::synchronize();
			CUDA::copyDeviceToHost(&hostFlag, deviceFlag, 1);

			if (storeData) {
				CUDA::copyDeviceToHost(intermediateResult, deviceCoverageMap, numElements);
				std::string fileName = "iteration_" + std::to_string(iterations) + "." + std::to_string(innerIterations);
				IO::writeCUDAPairMatrix(intermediateResult, rows, cols, fileName);
				innerIterations++;
			}
		} while (hostFlag);

		CUDA::set(deviceFlag, false);
		EEDT << <blocksPerGrid, threadsPerBlock >> > (deviceBoundary, deviceCoverageMap, deviceFlag, rows, cols, config.serviceRadius);
		CUDA::synchronize();
		CUDA::copyDeviceToHost(&hostFlag, deviceFlag, 1);

		if (storeData) {
			CUDA::copyDeviceToHost(intermediateResult, deviceCoverageMap, numElements);

			std::string fileName = "iteration_" + std::to_string(iterations) + "." + std::to_string(innerIterations);
			IO::writeCUDAPairMatrix(intermediateResult, rows, cols, fileName);
		}
		iterations++;

	} while (hostFlag);

	config.solutionData += "Total iterations: " + std::to_string(iterations) + "\n\n";
	
	MapElement* hostCoverageMap = new MapElement[numElements];
	CUDA::copyDeviceToHost(hostCoverageMap, deviceCoverageMap, numElements);

	CUDA::free(deviceBoundary);
	CUDA::free(deviceCoverageMap);
	CUDA::free(deviceFlag);

	return hostCoverageMap;
}
