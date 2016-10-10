
cmake_minimum_required (VERSION 3.3)
project(kokkos_practice)

find_package(CUDA QUIET)

set(KOKKOS_PATH "/home/michael/Documents/Programming/kokkos")
option(CUDA_TARGET "Build for CUDA devices" OFF)
option(OPENMP_TARGET "Build for OpenMP devices" ON)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories("${KOKKOS_PATH}/core/src")

file(GLOB SRC "src/*.[ch]pp")
file(GLOB KOKKOS_SRC "${KOKKOS_PATH}/core/src/impl/*.cpp" "${KOKKOS_PATH}/core/src/Threads/*.cpp")

add_compile_options(-std=c++11 -Wall)

if(${CUDA_TARGET} AND ${CUDA_FOUND})
  set(CUDA_64_BIT_DEVICE_CODE ON)
  list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
  cuda_include_directories("${KOKKOS_PATH}/core")
  set(KCUDA_PATH "${KOKKOS_PATH}/core/src/OpenMP")
  cuda_include_directories("${KCUDA_PATH}")
  file(GLOB KCUDA_SRC "${KCUDA_PATH}/*.cpp")
  cuda_add_executable(kokkos_practice_cuda ${SRC} ${KOKKOS_SRC} ${KCUDA_SRC})
  target_link_libraries(kokkos_practice_cuda "-ldl -lpthread")
endif()

if(${OPENMP_TARGET})
  set(KOMP_PATH "${KOKKOS_PATH}/core/src/OpenMP")
  include_directories(${KOMP_PATH})
  file(GLOB KOMP_SRC "${KOMP_PATH}/*.cpp")
  add_compile_options("-fopenmp")
  add_executable(kokkos_practice_omp ${SRC} ${KOKKOS_SRC} ${KOMP_SRC})
  target_link_libraries(kokkos_practice_omp "-ldl -lpthread")
endif()
