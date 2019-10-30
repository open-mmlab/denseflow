cmake_minimum_required(VERSION 3.12)
project( denseFlow )

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

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


find_package( OpenCV 4.1.0 REQUIRED PATHS /mnt/lustrenvme/lizz/app/opencv/ )
find_package( LibZip REQUIRED )
find_package( HDF5 COMPONENTS HL REQUIRED )

# BOOST
FIND_PACKAGE(Boost REQUIRED python)
FIND_PACKAGE(PythonLibs REQUIRED)

if(LIBZIP_VERSION VERSION_LESS 0.11)
    #old version LibZip
    add_definitions(-DUSE_OBSEL_LIBZIP)
endif()

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

message(STATUS "HDF5 library status:")
message(STATUS "    config: ${HDF5_DIR}")
message(STATUS "    version: ${HDF5_VERSION}")
message(STATUS "    libraries: ${HDF5_LIBS}")
message(STATUS "    include path: ${HDF5_INCLUDE_DIRS}")

include_directories( ${OpenCV_INCLUDE_DIRS} ${LIBZIP_INCLUDE_DIR_ZIP} ${LIBZIP_INCLUDE_DIR_ZIPCONF} ${HDF5_INCLUDE_DIRS} ${HDF5_HL_INCLUDE_DIRS} include/ include/easylogging++/src include/CLUE/include)
include_directories(SYSTEM ${Boost_INCLUDE_DIR} ${PYTHON_INCLUDE_DIR})
link_directories(${OpenCV_LIB_DIRS} ${HDF5_LIB_DIRS} ${HDF5_HL_LIB_DIRS})

add_library(denseflow src/common.cpp src/dense_flow.cpp src/dense_flow_gpu.cpp src/dense_flow_frames_gpu.cpp src/dense_flow_video_gpu.cpp src/dense_warp_flow_gpu.cpp src/zip_utils.cpp src/hdf5_utils.cpp)
target_link_libraries( denseflow ${OpenCV_LIBS} ${LIBZIP_LIBRARY} ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})

add_library(pydenseflow SHARED src/py_denseflow.cpp)
target_link_libraries(pydenseflow
        denseflow
        ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${OpenCV_LIBS}
        )

# add_executable( extract_cpu tools/extract_flow.cpp)
# target_link_libraries( extract_cpu ${OpenCV_LIBS} ${LIBZIP_LIBRARY} denseflow)

# add_executable( extract_gpu tools/extract_flow_gpu.cpp)
# target_link_libraries( extract_gpu ${OpenCV_LIBS} ${LIBZIP_LIBRARY} denseflow)

# add_executable( extract_frames_gpu tools/extract_flow_frames_gpu.cpp)
# target_link_libraries( extract_frames_gpu ${OpenCV_LIBS} ${LIBZIP_LIBRARY} denseflow)

add_executable( extract_video_gpu tools/extract_flow_video_gpu.cpp)
target_link_libraries( extract_video_gpu ${OpenCV_LIBS} ${LIBZIP_LIBRARY} denseflow)

# add_executable( extract_warp_gpu tools/extract_warp_flow_gpu.cpp)
# target_link_libraries( extract_warp_gpu ${OpenCV_LIBS} ${LIBZIP_LIBRARY} denseflow)
