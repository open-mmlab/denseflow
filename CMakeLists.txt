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

find_package( OpenCV 4.1.2 REQUIRED PATHS /home/lizz/app )

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

include_directories( ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} include/ include/easylogging++/src include/CLUE/include)
link_directories(${OpenCV_LIB_DIRS} ${Boost_LIBRARIES} )

add_library(denseflow src/common.cpp src/utils.cpp src/dense_nv_flow_video_gpu.cpp )
target_link_libraries( denseflow ${OpenCV_LIBS} Boost::filesystem ${Boost_LIBRARIES} )

add_executable( extract_nvflow tools/extract_nv_flow_video_gpu.cpp)
target_link_libraries( extract_nvflow ${OpenCV_LIBS} Boost::filesystem ${Boost_LIBRARIES} denseflow)
