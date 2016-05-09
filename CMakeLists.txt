# Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
# Licensed under the MIT license. See the license file LICENSE.

project(mmf)
cmake_minimum_required(VERSION 2.8.1)
set(POD_NAME mmf)
include(cmake/pods.cmake)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)


set(CMAKE_COLOR_MAKEFILE ON)
IF(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug) # Release; debug; ...
  set(CMAKE_BUILD_TYPE Release) # Release; debug; ...
ENDIF()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "default install path" FORCE )
endif()

find_package (Eigen3 REQUIRED)
find_package (CUDA REQUIRED)
find_package (OpenMP)
#find_package (OpenCV REQUIRED)
#find_package (PCL REQUIRED)
#FIND_PACKAGE (Boost REQUIRED)

if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# add flags depending on build type
set(CMAKE_CXX_FLAGS_RELEASE "-g -Wall -Wno-deprecated -O2 -finline-functions -DNDEBUG")
#set(CMAKE_CXX_FLAGS_RELEASE "-g -Wall -Wno-deprecated -O2 -finline-functions")
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wno-deprecated")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=\"sm_35,compute_35\")

include_directories(
  include
  #./common/inc
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  #  /usr/include
#  $ENV{WORKSPACE_HOME}/research/global/include
  #/usr/include/pcl-1.7/
  #  /usr/include/ni
  #  /usr/include/vtk-5.8/
  ${Boost_INCLUDE_DIRS}
  )

link_directories(
  ${PCL_LIBRARY_DIRS}
  ${Boost_LIBRARY_DIRS}
#  $ENV{WORKSPACE_HOME}/research/global/lib
  #  $ENV{WORKSPACE_HOME}/3rdparty/pcl-pcl-1.7.1/build
  )

cuda_add_library(mmf SHARED
  #  ./src/realtimeMF.cpp
  ./src/optimizationSO3.cpp
  ./src/optimizationSO3_approx.cpp
  ./src/optimizationSO3_vmf.cpp
  ./src/optimizationSO3_vmfCF.cpp
  ./cuda/optimizationSO3_approx.cu
  ./cuda/optimizationSO3.cu
  # jsCOre
#  ./cuda/vectorSum_kernel.cu
#  ./cuda/labelMap_kernel.cu
#  ./cuda/copy_kernel.cu
  )
target_link_libraries(mmf)
#  dpMMlowVar)

#add_subdirectory(./librealtimeMF)

#INSTALL(TARGETS mmf #realtimeMF_openni
#  RUNTIME DESTINATION bin
#  LIBRARY DESTINATION lib
#  ARCHIVE DESTINATION lib
#  )
pods_install_libraries(mmf)
pods_install_headers(
  ./include/mmf/sphereSimple.hpp
  ./include/mmf/cudaSphereHelpers.h
  ./include/mmf/optimizationSO3.hpp
  ./include/mmf/optimizationSO3_vmf.hpp
  ./include/mmf/optimizationSO3_vmfCF.hpp
  ./include/mmf/optimizationSO3_approx.hpp
  ./include/mmf/root_includes.hpp
  DESTINATION mmf)

#install(DIRECTORY include/mmf/ DESTINATION include/mmf/
#  FILES_MATCHING PATTERN "*.h")
##install(DIRECTORY include/nvidia DESTINATION include/nvidia
##          FILES_MATCHING PATTERN "*.h")
#install(DIRECTORY include/mmf/ DESTINATION include/mmf/
#  FILES_MATCHING PATTERN "*.hpp")
#install(DIRECTORY include/jsCore/ DESTINATION include/jsCore/
#  FILES_MATCHING PATTERN "*.hpp")
#install(DIRECTORY include/nvidia/ DESTINATION include/nvidia/
#  FILES_MATCHING PATTERN "*.h")
#
