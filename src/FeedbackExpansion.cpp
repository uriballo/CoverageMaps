#include "FeedbackExpansion.h"

void runExactExpansion(config conf) {
	int rows, cols;

	bool* boundary = IO::extractImageBoundary(conf.domainPath, rows, cols);

	int numElements = rows * cols;
	
	/*
	float* hostBlockIDs = new float[numElements];
	float* deviceBlockIDs;

	CUDA::allocate(deviceBlockIDs, numElements);

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

	getBlockID<<<blocksPerGrid, threadsPerBlock>>>(deviceBlockIDs, rows, cols);
	CUDA::synchronize();

	CUDA::copyDeviceToHost(hostBlockIDs, deviceBlockIDs, numElements);

	IO::writeFloatMatrix(hostBlockIDs, rows, cols, "blockids");

	*/
	
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
		//UTILS::showRGB(boundary, sourceDistribution, coverageMap, conf.radius, rows, cols);
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
	cudaMalloc(&deviceCoverageMap, numElements * sizeof(CUDAPair<float, int>));
	cudaMemcpy(deviceCoverageMap, hostCoverageMap, numElements * sizeof(CUDAPair<float, int>), cudaMemcpyHostToDevice);
	//CUDA::allocateAndCopy(deviceCoverageMap, hostCoverageMap, numElements);

	bool* deviceBoundary;
	cudaMalloc(&deviceBoundary, numElements * sizeof(bool));
	cudaMemcpy(deviceBoundary, boundary, numElements * sizeof(bool), cudaMemcpyHostToDevice);
	//CUDA::allocateAndCopy(deviceBoundary, boundary, numElements);

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

	int iterations = 0;
	cudaMalloc(&deviceFlag, sizeof(bool));
	//CUDA::allocate(deviceFlag, 1);

	do {
		do {
			//CUDA::set(deviceFlag, false);
			cudaMemset(deviceFlag, false, sizeof(bool));
			euclideanExpansion << <blocksPerGrid, threadsPerBlock >> > (deviceBoundary, deviceCoverageMap, deviceFlag, rows, cols, radius);
			cudaDeviceSynchronize();
			//CUDA::synchronize();
			//CUDA::copyDeviceToHost(&hostFlag, deviceFlag, 1);
			cudaMemcpy(&hostFlag, deviceFlag, sizeof(bool), cudaMemcpyDeviceToHost);
		} while (hostFlag);

		iterations++;
		//CUDA::set(deviceFlag, false);
		cudaMemset(deviceFlag, false, sizeof(bool));

		EEDT << <blocksPerGrid, threadsPerBlock >> > (deviceBoundary, deviceCoverageMap, deviceFlag, rows, cols, radius);
		cudaDeviceSynchronize();
		//CUDA::synchronize();
		//CUDA::copyDeviceToHost(&hostFlag, deviceFlag, 1);
		cudaMemcpy(&hostFlag, deviceFlag, sizeof(bool), cudaMemcpyDeviceToHost);
	} while (hostFlag);
	
	std::cout << std::endl << "Iterations: " << iterations << std::endl;

	//CUDA::copyDeviceToHost(hostCoverageMap, deviceCoverageMap, numElements);
	cudaMemcpy(hostCoverageMap, deviceCoverageMap, numElements * sizeof(CUDAPair<float, int>), cudaMemcpyDeviceToHost);

	CUDA::free(deviceBoundary);
	CUDA::free(deviceCoverageMap);
	CUDA::free(deviceFlag);

	return hostCoverageMap;
}
