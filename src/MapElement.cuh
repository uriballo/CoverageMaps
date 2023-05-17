#pragma once 

//#ifndef MAP_ELEMENT_CUH
//#define MAP_ELEMENT_CUH

#include <utility>

class MapElement {
public:
    __host__ __device__ MapElement() : distance(), predecessor(), source() {}
    __host__ __device__ MapElement(float dist, int pred, int srcIdx) : distance(dist), predecessor(pred), source(srcIdx) {}

    __host__ __device__ MapElement(const MapElement& other) = default;
    __host__ __device__ MapElement(MapElement&& other) = default;

    __host__ __device__ MapElement& operator=(const MapElement& other) = default;
    __host__ __device__ MapElement& operator=(MapElement&& other) = default;

    __host__ __device__ ~MapElement() = default;

    static constexpr size_t size = sizeof(float) + sizeof(int) + sizeof(int);

    __host__ __device__ void* operator new[](size_t size) {
        void* ptr;
        cudaMallocManaged(&ptr, size);
        return ptr;
    }

    __host__ __device__ void operator delete[](void* ptr) {
        cudaFree(ptr);
    }

    float distance;
    int predecessor;
    int source;
};

__host__ __device__ __inline__ MapElement make_CUDATriplet(float dist, int pred, int srcIdx) {
    return MapElement(dist, pred, srcIdx);
}

__host__ __device__ __inline__ void* operator new(size_t size, MapElement& p, float dist, int pred, int srcIdx) {
    void* ptr;
    cudaMalloc(&ptr, sizeof(MapElement));
    p.distance = dist;
    p.predecessor = pred;
    p.source = srcIdx;
    return ptr;
}

__host__ __device__ __inline__ void operator delete(void* ptr, MapElement& p, float dist, int pred, int srcIdx) {
    cudaFree(ptr);
    p.distance = 0.0f;
    p.predecessor = 0;
    p.source = 0;
}

//#endif // MAP_ELEMENT_CUH
