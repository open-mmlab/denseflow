//
// Created by yjxiong on 11/18/15.
//

#ifndef DENSEFLOW_DENSE_FLOW_H
#define DENSEFLOW_DENSE_FLOW_H

#include "common.h"
#include "easylogging++.h"

void calcDenseNvFlowVideoGPU(path video_path, path output_dir, string algorithm, int step, int bound, int new_width,
                             int new_height, int new_short, int dev_id, bool verbose, Stream stream);

#endif // DENSEFLOW_DENSE_FLOW_H
