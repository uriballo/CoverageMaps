#pragma once


#include <string>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <random>
#include <filesystem>
#include "MapElement.cuh"
#include "CUDACommon.cuh"

struct SystemParameters {
	std::string imagePath;
	std::string imageName;

	bool storeBoundary;
	bool storeIterCoverage;

	bool customDistribution;
	std::string serviceDistribution;

	bool maxCoverage;

	std::string solutionData;
};

struct AlgorithmParameters {
	int numServices;
	float serviceRadius;
	bool useEuclideanExpansion;
	bool useExactExpansion;
};

struct OptimizationParameters {
	int numGenerations;
	int populationSize;
	float mutationRate;
	float stopThreshold;
};

#define CUDA_CHECK(x) checkCUDAError((x), __FILE__, __LINE__)

namespace CUDA {
	void inline checkCUDAError(cudaError_t result, const char* file, const int line) {
		if (result != cudaSuccess) {
			fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(result));
			exit(1);
		}
	}

	// Allocate memory on the GPU for an array of size elements
	template<typename T>
	void inline allocate(T*& devicePtr, int size) {
		CUDA_CHECK(cudaMalloc(&devicePtr, size * sizeof(T)));
	}

	// Copy host array to device array
	template<typename T>
	void inline copyHostToDevice(T* devicePtr, const T* hostPtr, int size) {
		CUDA_CHECK(cudaMemcpy(devicePtr, hostPtr, size * sizeof(T), cudaMemcpyHostToDevice));
	}

	// Copy device array to host array
	template<typename T>
	void inline copyDeviceToHost(T* hostPtr, const T* devicePtr, int size) {
		CUDA_CHECK(cudaMemcpy(hostPtr, devicePtr, size * sizeof(T), cudaMemcpyDeviceToHost));
	}

	template<typename T>
	void inline copyVarDeviceToHost(T* hostPtr, const T* devicePtr) {
		CUDA_CHECK(cudaMemcpy(hostPtr, devicePtr, sizeof(T), cudaMemcpyDeviceToHost));
	}

	// Free memory on the GPU for the specified pointer
	template<typename T>
	void inline free(T* devicePtr) {
		CUDA_CHECK(cudaFree(devicePtr));
	}

	// Synchronize the CPU with the GPU
	void inline sync() {
		CUDA_CHECK(cudaDeviceSynchronize());
	}

	// Allocate memory on the GPU for an array of size elements and set its contents to value
	template<typename T>
	void inline allocateAndSet(T*& devicePtr, int size, const T& value) {
		CUDA_CHECK(cudaMalloc(&devicePtr, size * sizeof(T)));
		CUDA_CHECK(cudaMemset(devicePtr, value, size * sizeof(T)));
	}

	template<typename T>
	void inline allocateAndCopy(T*& devicePtr, const T* hostPtr, int size) {
		CUDA_CHECK(cudaMalloc(&devicePtr, size * sizeof(T)));
		CUDA_CHECK(cudaMemcpy(devicePtr, hostPtr, size * sizeof(T), cudaMemcpyHostToDevice));
	}

	template<typename T>
	void inline set(T* devicePtr, const T& value) {
		CUDA_CHECK(cudaMemcpy(devicePtr, &value, sizeof(T), cudaMemcpyHostToDevice));
	}

	void inline allocateArray(cudaArray_t& array, const cudaChannelFormatDesc& desc, int width, int height) {
		CUDA_CHECK(cudaMallocArray(&array, &desc, width, height));
	}

	template<typename T>
	void inline copyToArray(cudaArray_t array, const T* hostPtr, size_t count) {
		CUDA_CHECK(cudaMemcpyToArray(array, 0, 0, hostPtr, count * sizeof(T), cudaMemcpyHostToDevice));
	}

	cudaTextureObject_t inline createTextureObject(cudaArray* domainArray) {
		cudaTextureObject_t texDomainObj = 0; // Initialize texture object

		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc)); // Initialize resource descriptor
		resDesc.resType = cudaResourceTypeArray; // Set resource type to array
		resDesc.res.array.array = domainArray; // Set resource array to the input domain array

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc)); // Initialize texture descriptor
		texDesc.addressMode[0] = cudaAddressModeClamp; // Set address mode in x-direction to clamp
		texDesc.addressMode[1] = cudaAddressModeClamp; // Set address mode in y-direction to clamp
		texDesc.filterMode = cudaFilterModePoint; // Set filter mode to point
		texDesc.readMode = cudaReadModeElementType; // Set read mode to element type
		texDesc.normalizedCoords = 0; // Set normalized coordinates to false

		CUDA_CHECK(cudaCreateTextureObject(&texDomainObj, &resDesc, &texDesc, NULL)); // Create texture object

		return texDomainObj; // Return created texture object
	}
}

namespace IO {
	/*
	  * @brief Reads an image file and returns it as a cv::Mat object.
	  * @param path The path to the image file.
	  * @param mode The mode in which the image should be loaded.
	  * @return The loaded image as a cv::Mat object.
	*/
	cv::Mat readImage(const std::string path, cv::ImreadModes mode);

	/*
	  * @brief Writes a boolean matrix to a file.
	  * @param mat The boolean matrix.
	  * @param rows The number of rows in the matrix.
	  * @param cols The number of columns in the matrix.
	  * @param fileName The name of the file to write to.
	  * @param path The path to the directory where the file should be written.
	  * @param extension The file extension.
	*/
	void writeBoolMatrix(bool* mat, int rows, int cols, std::string fileName, std::string path = "output/", std::string extension = ".txt");

	/*
	  * @brief Writes an integer matrix to a file.
	  * @param mat The integer matrix.
	  * @param rows The number of rows in the matrix.
	  * @param cols The number of columns in the matrix.
	  * @param fileName The name of the file to write to.
	  * @param path The path to the directory where the file should be written.
	  * @param extension The file extension.
	*/
	void writeIntMatrix(int* mat, int rows, int cols, std::string fileName, std::string path = "output/", std::string extension = ".txt");

	/*
	  * @brief Writes a float matrix to a file.
	  * @param mat The float matrix.
	  * @param rows The number of rows in the matrix.
	  * @param cols The number of columns in the matrix.
	  * @param fileName The name of the file to write to.
	  * @param path The path to the directory where the file should be written.
	  * @param extension The file extension.
	*/
	void writeFloatMatrix(float* mat, int rows, int cols, std::string fileName, std::string path = "output/", std::string extension = ".txt");

	/*
	  * @brief Writes a float matrix to a file.
	  * @param mat The float matrix.
	  * @param rows The number of rows in the matrix.
	  * @param cols The number of columns in the matrix.
	  * @param fileName The name of the file to write to.
	  * @param path The path to the directory where the file should be written.
	  * @param extension The file extension.
	*/
	void writeCoverageMap(const MapElement* distanceMap, int rows, int cols, std::string fileName, std::string path = "output/", std::string extension = ".txt");


	void writeRGBImage(const cv::Mat& imageRGB, std::string filePath);

	/*
	  * @brief Converts a grayscale cv::Mat image to a float array.
	  * @param image The grayscale image as a cv::Mat object.
	  * @return The converted float array.
	*/
	float* convertCV2Float(const cv::Mat& image);

	/*
	  * @brief Converts a float array to a grayscale cv::Mat image.
	  * @param array The float array.
	  * @param rows The number of rows in the image.
	  * @param cols The number of columns in the image.
	  * @return The converted grayscale image as a cv::Mat object.
	*/
	cv::Mat convertFloat2CV(const float* array, int rows, int cols);

	/*
	  * @brief Creates an RGB image from separate arrays of red, green, and blue channels.
	  * @param outputR The array containing the red channel values.
	  * @param outputG The array containing the green channel values.
	  * @param outputB The array containing the blue channel values.
	  * @param numRows The number of rows in the image.
	  * @param numCols The number of columns in the image.
	  * @return The created RGB image as a cv::Mat object.
	*/
	cv::Mat createRGBImage(float* outputR, float* outputG, float* outputB, int numRows, int numCols);
}

namespace UTILS {
	/*
	  * @brief Converts a string of comma-separated integers to a vector of integers.
	  * @param str The input string.
	  * @return The vector of integers.
	*/
	std::vector<int> convertString2IntVector(const std::string& str);

	/*
	  * @brief Converts a vector of integers to a string representation.
	  * @param vec The vector of integers.
	  * @return The string representation of the vector.
	*/
	std::string convertIntVector2String(const std::vector<int>& vec);

	/*
	  * @brief Generates a random distribution of points within a given boundary.
	  * @param boundary The boundary array.
	  * @param rows The number of rows in the boundary.
	  * @param cols The number of columns in the boundary.
	  * @param N The number of points to generate.
	  * @return The generated random distribution of points.
	*/
	std::vector<int> generateRandomDistribution(const int* boundary, int rows, int cols, int N);

	/*
	  * @brief Processes the results of coverage mapping and generates an RGB image.
	  * @param boundary The boundary array.
	  * @param coverageMap The coverage map array.
	  * @param sourceDistribution The distribution of source points.
	  * @param numSources The number of source points.
	  * @param radius The radius for processing.
	  * @param rows The number of rows in the coverage map.
	  * @param cols The number of columns in the coverage map.
	  * @return The generated RGB image.
	*/
	cv::Mat processResultsRGB(const int* boundary, const MapElement* coverageMap, const int* sourceDistribution, int numSources, const float radius, int rows, int cols);

	/*
	  * @brief Frees the resources used for expansion.
	  * @param domainTexture The domain texture object.
	  * @param domainArray The domain array.
	  * @param deviceCoverageMap The device coverage map.
	  * @param domain The domain array.
	*/
	void freeExpansionResources(cudaTextureObject_t domainTexture, cudaArray* domainArray, MapElement* deviceCoverageMap, int* domain);
}

