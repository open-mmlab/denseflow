//
// Created by yjxiong on 11/18/15.
//

#ifndef DENSEFLOW_COMMON_H_H
#define DENSEFLOW_COMMON_H_H

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

using namespace cv;
using std::string;
using std::vector;

void convertFlowToImage(const Mat &flow_x, const Mat &flow_y, Mat &img_x, Mat &img_y, double lowerBound,
                        double higherBound);
void drawOptFlowMap(const Mat &flow, Mat &cflowmap, int step, double, const Scalar &color);

void encodeFlowMap(const Mat &flow_map_x, const Mat &flow_map_y, std::vector<uchar> &encoded_x,
                   std::vector<uchar> &encoded_y, int bound, bool to_jpg = true);

inline void initializeMats(const Mat &frame, Mat &capture_image, Mat &capture_gray, Mat &prev_image, Mat &prev_gray) {
    capture_image.create(frame.size(), CV_8UC3);
    capture_gray.create(frame.size(), CV_8UC1);

    prev_image.create(frame.size(), CV_8UC3);
    prev_gray.create(frame.size(), CV_8UC1);
}

inline bool fileExists(const string &name) {
    struct stat info;
    return stat(name.c_str(), &info) == 0;
}

inline bool dirExists(const string &path) {
    struct stat info;
    return stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR);
}

void writeImages(std::vector<std::vector<uchar>> images, std::string name_prefix);
void writeFlowImages(std::vector<std::vector<uchar>> images, std::string name_prefix, const int step = 1);
void writeImagesV2(std::vector<std::vector<uchar>> images, std::vector<std::vector<uchar>> flow_x,
                   std::vector<std::vector<uchar>> flow_y, std::vector<std::string> names,
                   const string &output_root_dir);
#endif // DENSEFLOW_COMMON_H_H
