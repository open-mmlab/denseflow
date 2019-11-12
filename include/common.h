//
// Created by yjxiong on 11/18/15.
//

#ifndef DENSEFLOW_COMMON_H_H
#define DENSEFLOW_COMMON_H_H

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"

#include <boost/filesystem.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

using namespace cv;
using namespace cv::cuda;
using boost::filesystem::path;
using std::string;
using std::vector;

void convertFlowToImage(const Mat &flow_x, const Mat &flow_y, Mat &img_x, Mat &img_y, double lowerBound,
                        double higherBound);

void encodeFlowMap(const Mat &flow_map_x, const Mat &flow_map_y, vector<uchar> &encoded_x, vector<uchar> &encoded_y,
                   int bound, bool to_jpg = true);

void writeImages(vector<vector<uchar>> images, string name_prefix);

void writeFlowImages(vector<vector<uchar>> images, string name_prefix, const int step = 1, const int start = 0);

#endif // DENSEFLOW_COMMON_H_H
