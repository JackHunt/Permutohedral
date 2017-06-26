#Set CMake version and project name.
cmake_minimum_required(VERSION 2.8)
project(PermutohedralLattice)

#Set compiler flags for Windows and Unix, respectively.
if(MSVC_IDE)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-DUSING_CMAKE=1)
else(MSVC_IDE)
  set(CMAKE_CXX_FLAGS_DEBUG "-Wall -std=c++11 -fPIC -g -march=native")
  set(CMAKE_CXX_FLAGS_RELEASE "-std=c++11 -fPIC -O3 -march=native")
endif(MSVC_IDE)

#Try to find CUDA.
find_package(CUDA)
if(CUDA_FOUND)
  include_directories(${CUDA_INCLUDE_DIRS})
  set(WITH_CUDA TRUE)
  add_definitions(-DWITH_CUDA)
else(CUDA_FOUND)
  add_definitions(-DCPU_ONLY)
endif()

#Add sources.
set(CPU_HEADERS
  hash_table.hpp
  modified_permutohedral.hpp
)

set(CPU_SOURCES
  modified_permutohedral.cpp
)

set(CUDA_SOURCES
  modified_permutohedral.cu
  hash_helper.cu
)

#Add source groups for VS2015.
source_group(CPU FILES ${CPU_HEADERS} ${CPU_SOURCES})
source_group(CUDA FILES ${CUDA_SOURCES})

#Add library.
if(WITH_CUDA)
  cuda_add_library(Permutohedral ${CUDA_SOURCES} ${CPU_HEADERS} ${CPU_SOURCES})
else()
  add_library(Permutohedral ${CPU_HEADERS} ${CPU_SOURCES})
endif()