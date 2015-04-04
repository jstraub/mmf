# CMakeLists.txt 
project(adaptive_camera)

# required cmake version                                                                                                                                                                                   
cmake_minimum_required(VERSION 2.8)

# packages                                                                                                                                                                                                 

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake/cuda" ${CMAKE_MODULE_PATH})

find_package(CUDA QUIET REQUIRED)

set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)

set(BUILD_SHARED_LIBS OFF)

list(APPEND CUDA_NVCC_FLAGS -DBLAH="he he" -DTEST1="this is a test")

INCLUDE(FindMatlab)
ADD_DEFINITIONS(-fPIC)

set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
set(CUDA_64_BIT_DEVICE_CODE ON)

#set(BUILD_SHARED_LIBS OFF)

set(CUDA_SEPARABLE_COMPILATION ON)
# nvcc flags                                                                                                                                                                                               
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20 -Xcompiler -fPIC)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -Xcompiler -fPIC)
#SET_TARGET_PROPERTIES(${ADAPTIVE_CAMERA_O} PROPERTIES COMPILE_FLAGS "-fPIC")
#cuda_compile(ADAPTIVE_CAMERA_O adaptive_camera.cu)
#cuda_compile(SIMPLE_IMAGE_O SimpleImage.cu)
#cuda_compile(GPU_RANDOM_O gpu_random.cu)

#cuda_add_executable(driver ${HELPER_O} ${DRIVER_O})

#cuda_add_library(adaptive_camera SHARED ${ADAPTIVE_CAMERA_O} ${SIMPLE_IMAGE_O} )
cuda_add_library(adaptive_camera SHARED SimpleImage.cu adaptive_camera.cu)
cuda_add_library(normal_smoothing SHARED SimpleImage.cu normal_smoothing.cu)

install (TARGETS adaptive_camera normal_smoothing DESTINATION lib)

