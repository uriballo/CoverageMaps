#include "Coverage.h"

int* COVERAGE::getDomainFromImage(std::string filePath, int& rows, int& cols){
	// Read the image file into a matrix.
	cv::Mat image = IO::readImage(filePath, cv::IMREAD_GRAYSCALE);

	// Store some properties of the image in variables.
	rows = image.rows;
	cols = image.cols;
	int numElements = rows * cols;

	// Convert the matrix to a flat array of floats.
	float* hostImage = IO::convertCV2Float(image);

	int* deviceDomain;
	CUDA::allocate(deviceDomain, numElements);

	float* deviceImage;
	CUDA::allocateAndCopy(deviceImage, hostImage, numElements);

	// Set up the dimensions for the GPU kernel.
	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

	processImageAsDomain << <blocksPerGrid, threadsPerBlock >> > (deviceImage, deviceDomain, rows, cols);
	CUDA::sync();

	int* hostDomain = new int[numElements];
	CUDA::copyDeviceToHost(hostDomain, deviceDomain, numElements);

	CUDA::free(deviceDomain);
	CUDA::free(deviceImage);

	delete[] hostImage;

	return hostDomain;
}

cudaTextureObject_t COVERAGE::convertDomainToTexture(int* domain, cudaArray** domainArray, int rows, int cols){
	// Create channel format descriptor for integer type
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();

	// Allocate memory for the device array
	CUDA::allocateArray(*domainArray, channelDesc, cols, rows);

	// Copy host domain to device array
	CUDA::copyToArray(*domainArray, domain, rows * cols);

	return CUDA::createTextureObject(*domainArray);
}

MapElement* COVERAGE::getEmptyCoverageMap(int numElements)
{
	MapElement* deviceCoverageMap;

	CUDA::allocate(deviceCoverageMap, numElements);

	return deviceCoverageMap;
}

void COVERAGE::setupCoverageMap(MapElement* deviceCoverageMap, std::vector<int> servicesDistribution, float radius, int rows, int cols){
	int numElements = rows * cols;
	int numServices = servicesDistribution.size() /2;

	int* servicesArray = servicesDistribution.data();

	int* deviceServices;
	CUDA::allocateAndCopy(deviceServices, servicesArray, numServices * 2);

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

	initializeCoverageMap << <blocksPerGrid, threadsPerBlock >> > (deviceCoverageMap, radius + FLT_MIN, -1, deviceServices, numServices, numElements, cols);
	CUDA::sync();
}

void COVERAGE::computeExactCoverageMap(cudaTextureObject_t domain, MapElement* emptyCoverageMap, float radius, int rows, int cols){
	bool hostFlag = false;
	bool* deviceFlag;
	int numElements = rows * cols;

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

	CUDA::allocate(deviceFlag, 1);

	do {
		do {
			CUDA::set(deviceFlag, false);
			euclideanExpansion << <blocksPerGrid, threadsPerBlock >> > (domain, emptyCoverageMap, deviceFlag, rows, cols, radius);
			CUDA::sync();
			CUDA::copyDeviceToHost(&hostFlag, deviceFlag, 1);
		} while (hostFlag);

		CUDA::set(deviceFlag, false);
		EEDT << <blocksPerGrid, threadsPerBlock >> > (emptyCoverageMap, deviceFlag, rows, cols, radius);
		CUDA::sync();
		CUDA::copyDeviceToHost(&hostFlag, deviceFlag, 1);

	} while (hostFlag);

	CUDA::free(deviceFlag);
}

float COVERAGE::getCoveragePercent(cudaTextureObject_t domain, MapElement* coverageMap, int rows, int cols){
	int* deviceInteriorPoints;
	CUDA::allocateAndSet(deviceInteriorPoints, 1, 0);

	int* deviceCoveredPoints;
	CUDA::allocateAndSet(deviceCoveredPoints, 1, 0);

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

	evalCoverage << <blocksPerGrid, threadsPerBlock >> > (domain, coverageMap, rows, cols, deviceInteriorPoints, deviceCoveredPoints);
	CUDA::sync();

	int interiorPoints, coveredPoints;
	CUDA::copyDeviceToHost(&interiorPoints, deviceInteriorPoints, 1);
	CUDA::copyDeviceToHost(&coveredPoints, deviceCoveredPoints, 1);


	CUDA::free(deviceInteriorPoints);
	CUDA::free(deviceCoveredPoints);

	return 100 * (static_cast<float>(coveredPoints) / static_cast<float>(interiorPoints));
}
