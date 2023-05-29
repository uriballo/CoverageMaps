#pragma once

#include "CUDACommon.cuh"
#include <fstream>

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
