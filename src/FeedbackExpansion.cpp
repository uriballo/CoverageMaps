#include "FeedbackExpansion.h"
#include "GeneticAlgorithm.h"

float coveragePercent2(cudaTextureObject_t domainTexture, std::vector<int> genes, int numServices, float radius, MapElement* deviceCoverageMap, int rows, int cols) {
	deviceCoverageMap = initialCoverageMapGPU(genes, deviceCoverageMap, numServices, rows, cols, radius + FLT_MIN, -1);

	int* deviceInteriorPoints;
	CUDA::allocateAndSet(deviceInteriorPoints, 1, 0);

	int* deviceCoveredPoints;
	CUDA::allocateAndSet(deviceCoveredPoints, 1, 0);

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

	evalCoverage << <blocksPerGrid, threadsPerBlock >> > (domainTexture, deviceCoverageMap, rows, cols, deviceInteriorPoints, deviceCoveredPoints);
	CUDA::sync();

	int interiorPoints, coveredPoints;
	CUDA::copyDeviceToHost(&interiorPoints, deviceInteriorPoints, 1);
	CUDA::copyDeviceToHost(&coveredPoints, deviceCoveredPoints, 1);


	return 100* (static_cast<float>(coveredPoints) / static_cast<float>(interiorPoints));
}

std::vector<int> getServiceDistribution(const configuration& config, const int* domain, int rows, int cols) {
	if (config.customDistribution) {
		return UTILS::convertString2IntVector(config.serviceDistribution);
	}
	else {
		return UTILS::generateRandomDistribution(domain, rows, cols, config.numberOfServices);
	}
}

double computeCoverageMap(configuration& config, cudaTextureObject_t domainTexture, MapElement* deviceCoverageMap, int rows, int cols) {
	auto startTime = std::chrono::steady_clock::now();

	MapElement* coverageMap = computeCoverage(domainTexture, deviceCoverageMap, config, rows, cols);

	auto endTime = std::chrono::steady_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
	double seconds = duration.count() / 1000.0;

	return seconds;
}

void runExactExpansion(configuration& config) {
	int rows, cols;
	config.solutionData = "";

	OptimizationParameters opt;

	opt.mutationRate = .1;
	opt.numGenerations = 10;
	opt.populationSize = 50;
	opt.stopThreshold = 35.0;


	auto startTime = std::chrono::steady_clock::now();

	int* domain = IO::preProcessDomainImage(config.imagePath, rows, cols);
	cudaArray* domainArray;
	cudaTextureObject_t domainTexture = UTILS::convertDomain2Texture(domain, rows, cols, &domainArray);
	auto endTime = std::chrono::steady_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
	double seconds = duration.count() / 1000.0;

	config.solutionData += "Domain process time: " + std::to_string(seconds) + " (s)\n";

	int numElements = rows * cols;

	config.solutionData += "Domain dimensions: " + std::to_string(rows) + " x " + std::to_string(cols) + " (" + std::to_string(numElements) + " pixels) \n\n";
	
	MapElement* deviceCoverageMap;
	CUDA::allocate(deviceCoverageMap, numElements);

	MCLPSolver solver(domain, deviceCoverageMap, config.numberOfServices, rows, cols, config.serviceRadius, opt);
	
	startTime = std::chrono::steady_clock::now();
	Individual best =	solver.solve();
	endTime = std::chrono::steady_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
	seconds = duration.count() / 1000.0;
	config.solutionData += "Genetic Algorithm process time: " + std::to_string(seconds) + " (s)\n";

	if (config.storeBoundary) 
		IO::writeIntMatrix(domain, rows, cols, "domain");
	
	std::vector<int> servicesDistribution = best.genes; //getServiceDistribution(config, domain, rows, cols);

	deviceCoverageMap = initialCoverageMapGPU(servicesDistribution, config.numberOfServices, rows, cols, config.serviceRadius + FLT_MIN, -1);

	seconds = computeCoverageMap(config, domainTexture, deviceCoverageMap, rows, cols);

	config.solutionData += "Coverage map compute time: " + std::to_string(seconds) + " (s)\n";
	config.solutionData += "Coverage percent: " + std::to_string(best.fitness) + " %\n";

	cv::Mat processedResultsRGB = UTILS::processResultsRGB(domain, deviceCoverageMap, servicesDistribution.data(), config.numberOfServices, config.serviceRadius, rows, cols);

	IO::writeRGBImage(processedResultsRGB, "output/" + config.imageName);

	UTILS::freeExpansionResources(domainTexture, domainArray, deviceCoverageMap, domain);
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
	CUDA::sync();

	return deviceCoverageMap;
}

MapElement* initialCoverageMapGPU(std::vector<int> servicesDistribution, MapElement* deviceCoverageMap, int numServices, int rows, int cols, float initRadius, int initPredecessor) {
	int numElements = rows * cols;

	int* servicesArray = servicesDistribution.data();

	int* deviceServices;
	CUDA::allocateAndCopy(deviceServices, servicesArray, numServices * 2);

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

	initCoverageMap << <blocksPerGrid, threadsPerBlock >> > (deviceCoverageMap, initRadius, initPredecessor, deviceServices, numServices, numElements, cols);
	CUDA::sync();

	return deviceCoverageMap;
}

MapElement* computeCoverage(cudaTextureObject_t domainTexture, MapElement* deviceCoverageMap, configuration& config, int rows, int cols) {
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
			euclideanExpansion << <blocksPerGrid, threadsPerBlock >> > (domainTexture, deviceCoverageMap, deviceFlag, rows, cols, config.serviceRadius);
			CUDA::sync();
			CUDA::copyDeviceToHost(&hostFlag, deviceFlag, 1);

			if (storeData) {
				CUDA::copyDeviceToHost(intermediateResult, deviceCoverageMap, numElements);
				std::string fileName = "iteration_" + std::to_string(iterations) + "." + std::to_string(innerIterations);
				IO::writeCoverageMap(intermediateResult, rows, cols, fileName);
				innerIterations++;
			}
		} while (hostFlag);

		CUDA::set(deviceFlag, false);
		EEDT << <blocksPerGrid, threadsPerBlock >> > (deviceCoverageMap, deviceFlag, rows, cols, config.serviceRadius);
		CUDA::sync();
		CUDA::copyDeviceToHost(&hostFlag, deviceFlag, 1);

		if (storeData) {
			CUDA::copyDeviceToHost(intermediateResult, deviceCoverageMap, numElements);

			std::string fileName = "iteration_" + std::to_string(iterations) + "." + std::to_string(innerIterations);
			IO::writeCoverageMap(intermediateResult, rows, cols, fileName);
		}
		iterations++;

	} while (hostFlag );

	config.solutionData += "Total iterations: " + std::to_string(iterations) + "\n\n";
	
	MapElement* hostCoverageMap = new MapElement[numElements];
	CUDA::copyDeviceToHost(hostCoverageMap, deviceCoverageMap, numElements);

	//CUDA::free(deviceCoverageMap);
	CUDA::free(deviceFlag);

	return hostCoverageMap;
}
