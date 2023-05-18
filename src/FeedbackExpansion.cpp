#include "FeedbackExpansion.h"

cudaTextureObject_t getDomainGPU(const int* hostDomain, int rows, int cols, cudaArray** domainArray) {
	// Create a CUDA array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();

	CUDA::allocateArray(*domainArray, channelDesc, cols, rows);
//	cudaMallocArray(domainArray, &channelDesc, cols, rows);
//	cudaMemcpyToArray(*domainArray, 0, 0, hostDomain, rows * cols * sizeof(int), cudaMemcpyHostToDevice);
	CUDA::copyToArray(*domainArray, hostDomain, rows * cols);

	// Create a texture object
	cudaTextureObject_t texDomainObj = 0;
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = *domainArray;

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;

	cudaCreateTextureObject(&texDomainObj, &resDesc, &texDesc, NULL);

	return texDomainObj;
}

void runExactExpansion(configuration& config) {
	int rows, cols;
	int* domain = IO::preProcessDomainImage(config.imagePath, rows, cols);
	int numElements = rows * cols;

	config.solutionData = "";
	config.solutionData += "Domain dimensions: " + std::to_string(rows) + " x " + std::to_string(cols) + " (" + std::to_string(numElements) + " pixels) \n\n";
	
	if (config.storeBoundary) {
		IO::writeIntMatrix(domain, rows, cols, "domain");
	}

	std::vector<int> servicesDistribution;

	if (config.customDistribution) {
		servicesDistribution = UTILS::convertString2IntVector(config.serviceDistribution);
	}
	else {
		servicesDistribution = UTILS::generateRandomDistribution(domain, rows, cols, config.numberOfServices);
	}


	cudaArray* domainArray;
	cudaTextureObject_t texDomainObj = getDomainGPU(domain, rows, cols, &domainArray);

	MapElement* deviceCoverageMap = initialCoverageMapGPU(servicesDistribution, config.numberOfServices, rows, cols, config.serviceRadius + FLT_MIN, -1);

	auto startTime = std::chrono::steady_clock::now();

	MapElement* coverageMap = computeCoverage(texDomainObj, deviceCoverageMap, config, rows, cols);// computeCoverage(domain, servicesDistribution, config, rows, cols);

	auto endTime = std::chrono::steady_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
	double seconds = duration.count() / 1000.0;

	config.solutionData += "Coverage map compute time: " + std::to_string(seconds) + " (s)\n";

	cv::Mat processedResultsRGB = UTILS::processResultsRGB(domain, coverageMap, servicesDistribution.data(), config.numberOfServices, config.serviceRadius, rows, cols);
	IO::writeRGBImage(processedResultsRGB, "output/" + config.imageName);

	cudaDestroyTextureObject(texDomainObj);
	cudaFreeArray(domainArray);

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
	CUDA::sync();

	return deviceCoverageMap;
}

int* getDomainGPU(const int* hostDomain, int numElements) {
	int* deviceDomain;
	CUDA::allocateAndCopy(deviceDomain, hostDomain, numElements);

	return deviceDomain;
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

	} while (hostFlag);

	config.solutionData += "Total iterations: " + std::to_string(iterations) + "\n\n";
	
	MapElement* hostCoverageMap = new MapElement[numElements];
	CUDA::copyDeviceToHost(hostCoverageMap, deviceCoverageMap, numElements);

	CUDA::free(deviceCoverageMap);
	CUDA::free(deviceFlag);

	return hostCoverageMap;
}
