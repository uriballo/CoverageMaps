#ifndef CUDA_PAIR_H
#define CUDA_PAIR_H

#include <utility>

template <typename T, typename U>
class CUDAPair {
public:
	__host__ __device__ CUDAPair() : first(), second() {}
	__host__ __device__ CUDAPair(const T& t, const U& u) : first(t), second(u) {}

	template <typename V, typename W>
	__host__ __device__ CUDAPair(const CUDAPair<V, W>& other) : first(other.first), second(other.second) {}

	__host__ __device__ CUDAPair(const CUDAPair& other) = default;
	__host__ __device__ CUDAPair(CUDAPair&& other) = default;

	__host__ __device__ CUDAPair& operator=(const CUDAPair& other) = default;
	__host__ __device__ CUDAPair& operator=(CUDAPair&& other) = default;

	__host__ __device__ ~CUDAPair() = default;

	static constexpr size_t size = sizeof(T) + sizeof(U);

	// Overload the new[] operator for CUDAPair
	__host__ __device__ void* operator new[](size_t size) {
		void* ptr;
		cudaMallocManaged(&ptr, size);
		return ptr;
	}

		// Overload the delete[] operator for CUDAPair
		__host__ __device__ void operator delete[](void* ptr) {
		cudaFree(ptr);
	}
	T first;
	U second;
};

template <typename T, typename U>
__host__ __device__ CUDAPair<T, U> make_CUDAPair(T t, U u) {
	return CUDAPair<T, U>(t, u);
}

template <typename T1, typename U1, typename T2, typename U2>
__host__ __device__ bool operator==(const CUDAPair<T1, U1>& lhs, const CUDAPair<T2, U2>& rhs) {
	return lhs.first == rhs.first && lhs.second == rhs.second;
}

template <typename T1, typename U1, typename T2, typename U2>
__host__ __device__ bool operator!=(const CUDAPair<T1, U1>& lhs, const CUDAPair<T2, U2>& rhs) {
	return !(lhs == rhs);
}

template <typename T1, typename U1, typename T2, typename U2>
__host__ __device__ bool operator<(const CUDAPair<T1, U1>& lhs, const CUDAPair<T2, U2>& rhs) {
	return lhs.first < rhs.first || (!(rhs.first < lhs.first) && lhs.second < rhs.second);
}

template <typename T1, typename U1, typename T2, typename U2>
__host__ __device__ bool operator>(const CUDAPair<T1, U1>& lhs, const CUDAPair<T2, U2>& rhs) {
	return rhs < lhs;
}

template <typename T1, typename U1, typename T2, typename U2>
__host__ __device__ bool operator<=(const CUDAPair<T1, U1>& lhs, const CUDAPair<T2, U2>& rhs) {
	return !(rhs < lhs);
}

template <typename T1, typename U1, typename T2, typename U2>
__host__ __device__ bool operator>=(const CUDAPair<T1, U1>& lhs, const CUDAPair<T2, U2>& rhs) {
	return !(lhs < rhs);
}

template <typename T, typename U>
void* operator new(size_t size, CUDAPair<T, U>& p, const T& t, const U& u) {
	void* ptr;
	cudaMalloc(&ptr, sizeof(CUDAPair<T, U>));
	p.first = t;
	p.second = u;
	return ptr;
}

template <typename T, typename U>
void operator delete(void* ptr, CUDAPair<T, U>& p, const T& t, const U& u) {
	cudaFree(ptr);
	p.first = T();
	p.second = U();
}

#endif // CUDAPair_H
