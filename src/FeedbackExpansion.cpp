#include "FeedbackExpansion.h"

void runExactExpansion(config conf) {
	int rows, cols;

	bool* boundary = IO::extractImageBoundary(conf.domainPath, rows, cols);

	int numElements = rows * cols;
	
	if (conf.verboseMode) {
		IO::writeBoolMatrix(boundary, rows, cols, "boundary");
	}

	std::cout << "Domain dimensions: (" << cols << " x " << rows << ")" << std::endl;

	int* sourceDistribution = conf.randomSources ?
		UTILS::getRandomSourceDistribution(boundary, rows, cols, conf.numSources)
		: conf.sources;

//	int sourceDistribution[] = { 430, 527 };

	CUDAPair<float, int>* coverageMap = computeCoverage(boundary, sourceDistribution, conf.radius, rows, cols, conf.numSources, conf.storeIterationContent);

	if (conf.storeFinalResult) {
		IO::writeCUDAPairMatrix(coverageMap, rows, cols, conf.outputFileName + "_result");
	}

	if (conf.showResults) {
	//	float* processedResult = UTILS::processResults_(boundary, coverageMap, conf.radius, rows, cols);
		
		cv::Mat processedResultsRGB = UTILS::processResultsRGB(boundary, sourceDistribution, coverageMap, conf.radius, rows, cols, conf.numSources);

	//	cv::Mat map = IO::floatToCV(processedResult, rows, cols);
		
	//	IO::storeBWImage(map, conf.outputFileName);
		IO::storeRGB(processedResultsRGB, "output/" + conf.outputFileName + ".png");

	//	delete[] processedResult;
	}

	delete[] coverageMap;
	delete[] boundary;
	delete[] sourceDistribution;
}

CUDAPair<float, int>* computeCoverage(const bool* boundary, const int* sourceDistribution, float radius, int rows, int cols, int numSources, bool storeIters) {
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
//	cudaMalloc(&deviceBoundary, numElements * sizeof(bool));
//	cudaMemcpy(deviceBoundary, boundary, numElements, cudaMemcpyHostToDevice);
	CUDA::allocateAndCopy(deviceBoundary, boundary, numElements);

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

	int iterations = 1;
	CUDA::allocate(deviceFlag, 1);
	CUDAPair<float, int>* intermediateResult = new CUDAPair<float, int>[numElements];

	do {
		int innerIterations = 1;

		do {
			CUDA::set(deviceFlag, false);
			euclideanExpansion << <blocksPerGrid, threadsPerBlock >> > (deviceBoundary, deviceCoverageMap, deviceFlag, rows, cols, radius);
			CUDA::synchronize();
			CUDA::copyDeviceToHost(&hostFlag, deviceFlag, 1);

			if (storeIters) {
				CUDA::copyDeviceToHost(intermediateResult, deviceCoverageMap, numElements);
				std::string fileName = "iteration_" + std::to_string(iterations) + "." + std::to_string(innerIterations);
				IO::writeCUDAPairMatrix(intermediateResult, rows, cols, fileName);
				innerIterations++;
			}
		} while (hostFlag);


		CUDA::set(deviceFlag, false);

		EEDT << <blocksPerGrid, threadsPerBlock >> > (deviceBoundary, deviceCoverageMap, deviceFlag, rows, cols, radius);
		CUDA::synchronize();
		CUDA::copyDeviceToHost(&hostFlag, deviceFlag, 1);
		if (storeIters) {
			CUDA::copyDeviceToHost(intermediateResult, deviceCoverageMap, numElements);

			std::string fileName = "iteration_" + std::to_string(iterations) + "." + std::to_string(innerIterations);
			IO::writeCUDAPairMatrix(intermediateResult, rows, cols, fileName);
		}
		iterations++;

	} while (hostFlag);
	
	//std::cout << std::endl << "Iterations: " << iterations << std::endl;

	CUDA::synchronize();

	CUDA::copyDeviceToHost(hostCoverageMap, deviceCoverageMap, numElements);

	//CUDAPair<float, int>* ids = new CUDAPair<float, int>[numElements];
	//testKernel << <blocksPerGrid, threadsPerBlock >> > (boundary, deviceCoverageMap, rows, cols, radius);
//	CUDA::synchronize();
//	CUDA::copyDeviceToHost(ids, deviceCoverageMap, numElements);

//	IO::writeCUDAPairMatrix(ids, rows, cols, "000ids");

	CUDA::free(deviceBoundary);
	CUDA::free(deviceCoverageMap);
	CUDA::free(deviceFlag);

	return hostCoverageMap;
}
