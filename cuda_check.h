#ifndef PERMUTOHEDRAL_CUDA_CHECK
#define PERMUTOHEDRAL_CUDA_CHECK

#include "cuda.h"
#include <cuda_runtime.h>
#include <iostream>

namespace Permutohedral {
#define CUDA_CHECK(ans){cudaAssert((ans), __FILE__, __LINE__);}
	void cudaAssert(cudaError_t code, const char *file, int line) {
		if (code != cudaSuccess) {
			std::cerr << "CUDA ERROR: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
			exit(code);
		}
	}

#define CUDA_POST_KERNEL_CHECK

}
#endif