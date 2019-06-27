//
// Created by yjxiong on 11/18/15.
//

#ifndef DENSEFLOW_DENSE_FLOW_H
#define DENSEFLOW_DENSE_FLOW_H

#include "common.h"
#include "easylogging++.h"

void calcDenseFlow(string file_name, int bound, int type, int step,
                   vector<vector<uchar> >& output_x,
                   vector<vector<uchar> >& output_y,
                   vector<vector<uchar> >& output_img);
void calcDenseFlowGPU(string file_name, int bound, int type, int step, int dev_id,
                      vector<vector<uchar> >& output_x,
                      vector<vector<uchar> >& output_y,
                      vector<vector<uchar> >& output_img,
                      int new_width=0, int new_height=0);

void calcDenseFlowPureGPU(std::string file_name, int bound, int type, int step, int dev_id,
                      std::vector<std::vector<uchar> >& output_x,
                      std::vector<std::vector<uchar> >& output_y,
                      std::vector<std::vector<uchar> >& output_img);

void calcDenseWarpFlowGPU(std::string file_name, int bound, int type, int step, int dev_id,
                          std::vector<std::vector<uchar> >& output_x,
                          std::vector<std::vector<uchar> >& output_y);

void MatchFromFlow_copy(const Mat& prev_grey, const Mat& flow_x, const Mat& flow_y, std::vector<Point2f>& prev_pts, std::vector<Point2f>& pts, const Mat& mask);

#endif //DENSEFLOW_DENSE_FLOW_H
