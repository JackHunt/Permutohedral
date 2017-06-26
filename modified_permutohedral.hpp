#ifndef PERMUTOHEDRAL_MODIFIED_PERMUTOHEDRAL_HPP_
#define PERMUTOHEDRAL_MODIFIED_PERMUTOHEDRAL_HPP_

#include <cstdlib>
#include <vector>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <cmath>
#include "hash_table.hpp"

#ifndef CPU_ONLY
#include "cuda_check.h"
#endif

/************************************************/
/***          ModifiedPermutohedral Lattice   ***/
/************************************************/

namespace Permutohedral {

typedef struct MatrixEntry {
  int index;
  float weight;
} MatrixEntry;

class ModifiedPermutohedral
{
protected:
  struct Neighbors{
    int n1, n2;
    Neighbors( int n1=0, int n2=0 ):n1(n1),n2(n2){
    }
  };

  // Check if GPU hash table if initialize
  bool is_init;

  std::vector<int> offset_, rank_;
  std::vector<float> barycentric_;
  std::vector<Neighbors> blur_neighbors_;

  // GPU specific
  MatrixEntry *matrix;
  HashTable table;


  // Number of elements, size of sparse discretized space, dimension of features width and height
  int N_, M_, d_, w_, h_;

  void init_cpu(const float* features, int num_dimensions, int num_points);
  void init_gpu(const float* features, int num_dimensions, int w, int h);

  void compute_cpu(float* out, const float* in, int value_size, bool reverse = false, bool add = false) const;
  void compute_cpu(double* out, const double* in, int value_size, bool reverse = false, bool add = false) const;

  void compute_gpu(float* out, const float* in, int value_size, bool reverse = false, bool add = false) const;
  void compute_gpu(double* out, const double* in, int value_size, bool reverse = false, bool add = false) const;

  void sseCompute(float* out, const float* in, int value_size, bool reverse = false, bool add = false) const;
  void sseCompute(double* out, const double* in, int value_size, bool reverse = false, bool add = false) const;

  void seqCompute(float* out, const float* in, int value_size, bool reverse = false, bool add = false) const;
  void seqCompute(double* out, const double* in, int value_size, bool reverse = false, bool add = false) const;


public:
  ModifiedPermutohedral();
  ~ModifiedPermutohedral(){
  #ifndef CPU_ONLY
    if(is_init)
      CUDA_CHECK(cudaFree(matrix));
  #endif
  }

  void init (const float* features, int num_dimensions, int w, int h, bool useGPU = false){
    if (!useGPU) {
      init_cpu(features, num_dimensions, w*h);
    }else{
#ifndef CPU_ONLY
      init_gpu(features, num_dimensions, w, h);
      is_init = true;
#else
      throw std::runtime_error("Compiled without CUDA support!");
#endif
    }
  }

  void compute(float* out, const float* in, int value_size, bool reverse = false, 
                bool add = false, bool useGPU = false) const{
    if (!useGPU) {
      compute_cpu(out, in, value_size, reverse, add);
    }else{
#ifndef CPU_ONLY
      compute_gpu(out, in, value_size, reverse, add);
#else
      throw std::runtime_error("Compiled without CUDA support!");
#endif
    }

  }
  void compute(double* out, const double* in, int value_size, bool reverse = false, 
                bool add = false, bool useGPU = false) const{
    if (!useGPU) {
      compute_cpu(out, in, value_size, reverse, add);
    }else{
#ifndef CPU_ONLY
      compute_gpu(out, in, value_size, reverse, add);
#else
      throw std::runtime_error("Compiled without CUDA support!");
#endif
    }
  }
};
}//namespace Permutohedral
#endif //PERMUTOHEDRAL_MODIFIED_PERMUTOHEDRAL_HPP_
