cmake_minimum_required(VERSION 3.12)
project(DenseFlow)

option(USE_HDF5 "Whether to build hdf5 for optical flow image saving" ON)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

find_package(CUDA REQUIRED)
find_package( OpenCV REQUIRED)

set(DenseFlow_LIB denseflow)

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

find_package(Boost REQUIRED COMPONENTS date_time filesystem iostreams)
message(STATUS "Boost library status:")
message(STATUS "    version: ${Boost_VERSION}")
message(STATUS "    libraries: ${Boost_LIBRARIES}")
message(STATUS "    include path: ${Boost_INCLUDE_DIRS}")

add_library(${DenseFlow_LIB} src/common.cpp src/utils.cpp src/dense_nv_flow_video_gpu.cpp)
include_directories(${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} include ${PROJECT_BINARY_DIR})
link_directories(${OpenCV_LIB_DIRS})
target_link_libraries(${DenseFlow_LIB} ${OpenCV_LIBS} Boost::filesystem ${Boost_LIBRARIES} pthread)

if (USE_HDF5)
    find_package(HDF5 COMPONENTS HL REQUIRED)
    message(STATUS "HDF5 library status:")
    message(STATUS "    version: ${HDF5_VERSION}")
    message(STATUS "    libraries: ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES}")
    message(STATUS "    include path: ${HDF5_INCLUDE_DIRS} ${HDF5_HL_INCLUDE_DIR}")
    set(HDF5_DEFINITION 1)
    include_directories(${HDF5_INCLUDE_DIRS} ${HDF5_HL_INCLUDE_DIR})
    target_link_libraries(${DenseFlow_LIB} ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})
else()
    set(HDF5_DEFINITION 0) 
endif()

configure_file(
    "${PROJECT_SOURCE_DIR}/include/config.h.in"
    "${PROJECT_BINARY_DIR}/config.h"
)

add_executable(extract_nvflow tools/extract_nv_flow_video_gpu.cpp)
target_link_libraries(extract_nvflow ${DenseFlow_LIB})
