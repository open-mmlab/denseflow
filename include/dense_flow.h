//
// Created by yjxiong on 11/18/15.
//

#ifndef DENSEFLOW_DENSE_FLOW_H
#define DENSEFLOW_DENSE_FLOW_H

#include "common.h"

void calcDenseFlow(string file_name, int bound, int type, int step,
                   vector<vector<uchar> >& output_x,
                   vector<vector<uchar> >& output_y,
                   vector<vector<uchar> >& output_img);
void calcDenseFlowGPU(string file_name, int bound, int type, int step, int dev_id,
                      vector<vector<uchar> >& output_x,
                      vector<vector<uchar> >& output_y,
                      vector<vector<uchar> >& output_img);

void calcDenseFlowPureGPU(string file_name, int bound, int type, int step, int dev_id,
                      vector<vector<uchar> >& output_x,
                      vector<vector<uchar> >& output_y,
                      vector<vector<uchar> >& output_img);

#endif //DENSEFLOW_DENSE_FLOW_H
