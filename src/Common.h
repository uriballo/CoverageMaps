#pragma once

#ifndef COMMON_H
#define COMMON_H

#include <string>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <random>
#include "CUDACommon.cuh"

struct config {
	std::string domainPath;

	float radius;

	int numSources;
	int* sources;
	bool randomSources;

	bool showResults;
	bool displayHeatMap;
};

#define CUDA_CHECK(x) checkCUDAError((x), __FILE__, __LINE__)

namespace IO {
	bool* extractImageBoundary(const std::string filePath, int& rows, int& cols);

	cv::Mat readImage(const std::string path, cv::ImreadModes mode);

	float* cvBW2FloatArray(const cv::Mat& image);

	void writeBoolMatrix(bool* mat, int rows, int cols, std::string fileName, std::string path = "output/", std::string extension = ".txt");

	void writeFloatMatrix(float* mat, int rows, int cols, std::string fileName, std::string path = "output/", std::string extension = ".txt");

	cv::Mat floatToCV(const float* array, int rows, int cols);

	void showBW(const cv::Mat& mat, std::string windowTitle);

	void showHeatMap(const float* heatMap, int rows, int cols);
}

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
	void inline synchronize() {
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
}

namespace UTILS {
	int* getRandomSourceDistribution(bool* boundary, int rows, int cols, int N);

	float* processResults(const bool* boundary, const float* coverageMap, const float radius, int rows, int cols);

	float* processResults_(const bool* boundary, const CUDAPair<float, int>* coverageMap, const float radius, int rows, int cols);

	void initializeCoverageMap(CUDAPair<float, int>* coverageMap, const float initDist, const int initPredecessor, const int size);

	void initializeSources(CUDAPair<float, int>* coverageMap, const int* sourceDistribution, const int numSources, const int cols);
}

#endif


