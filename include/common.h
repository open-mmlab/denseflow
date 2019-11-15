#ifndef DENSEFLOW_COMMON_H_H
#define DENSEFLOW_COMMON_H_H
#include "config.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#if (USE_HDF5)
    #include "hdf5.h"
    #include "hdf5_hl.h"
#endif
#include <boost/filesystem.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

using namespace cv;
using namespace cv::cuda;
using boost::filesystem::path;
using boost::filesystem::create_directories;
using boost::filesystem::exists;
using boost::filesystem::is_directory;
using std::string;
using std::vector;
using std::endl;
using std::cout;
using std::condition_variable;
using std::mutex;
using std::queue;
using std::thread;
using std::unique_lock;

void convertFlowToImage(const Mat &flow_x, const Mat &flow_y, Mat &img_x, Mat &img_y, double lowerBound,
                        double higherBound);

void encodeFlowMap(const Mat &flow_map_x, const Mat &flow_map_y, vector<uchar> &encoded_x, vector<uchar> &encoded_y,
                   int bound, bool to_jpg = true);

void writeImages(vector<vector<uchar>> images, string name_prefix);

void writeFlowImages(vector<vector<uchar>> images, string name_prefix, const int step = 1, const int start = 0);
#if (USE_HDF5)
void writeHDF5(const vector<Mat>& images, string name_prefix, string phase, const int step = 1, const int start = 0);
#endif
#endif // DENSEFLOW_COMMON_H_H
