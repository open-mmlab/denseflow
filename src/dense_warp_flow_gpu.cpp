#include "common.h"
#include "dense_flow.h"

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/gpu/gpu.hpp"

#include <stdio.h>
#include <iostream>

#include "warp_flow.h"

using namespace cv;
using namespace cv::gpu;

void calcDenseWarpFlowGPU(string file_name, int bound, int type, int step, int dev_id,
					  vector<vector<uchar> >& output_x,
					  vector<vector<uchar> >& output_y){
	VideoCapture video_stream(file_name);
	CHECK(video_stream.isOpened())<<"Cannot open video stream \""
								  <<file_name
								  <<"\" for optical flow extraction.";

	SurfFeatureDetector detector_surf(200);
	SurfDescriptorExtractor extractor_surf(true, true);
	std::vector<Point2f> prev_pts_flow, pts_flow;
	std::vector<Point2f> prev_pts_surf, pts_surf;
	std::vector<Point2f> prev_pts_all, pts_all;
	std::vector<KeyPoint> prev_kpts_surf, kpts_surf;
	Mat prev_desc_surf, desc_surf;

	setDevice(dev_id);
	Mat capture_frame, capture_image, prev_image, capture_gray, prev_gray, human_mask;
	Mat flow_x, flow_y;

	GpuMat d_frame_0, d_frame_1;
	GpuMat d_flow_x, d_flow_y;

	FarnebackOpticalFlow alg_farn;
	OpticalFlowDual_TVL1_GPU alg_tvl1;
	BroxOpticalFlow alg_brox(0.197f, 50.0f, 0.8f, 10, 77, 10);

	bool initialized = false;
	int cnt = 0;
	while(true){

		//build mats for the first frame
		if (!initialized){
			video_stream >> capture_frame;
			if (capture_frame.empty()) return; // read frames until end
			initializeMats(capture_frame, capture_image, capture_gray,
						   prev_image, prev_gray);
			capture_frame.copyTo(prev_image);
			cvtColor(prev_image, prev_gray, CV_BGR2GRAY);

			//detect key points
			human_mask = Mat::ones(capture_frame.size(), CV_8UC1);
			detector_surf.detect(prev_gray, prev_kpts_surf, human_mask);
			extractor_surf.compute(prev_gray, prev_kpts_surf, prev_desc_surf);

			initialized = true;
			for(int s = 0; s < step; ++s){
				video_stream >> capture_frame;
				cnt ++;
				if (capture_frame.empty()) return; // read frames until end
			}
		}else {
			capture_frame.copyTo(capture_image);
			cvtColor(capture_image, capture_gray, CV_BGR2GRAY);
			d_frame_0.upload(prev_gray);
			d_frame_1.upload(capture_gray);

			switch(type){
				case 0: {
					alg_farn(d_frame_0, d_frame_1, d_flow_x, d_flow_y);
					break;
				}
				case 1: {
					alg_tvl1(d_frame_0, d_frame_1, d_flow_x, d_flow_y);
					break;
				}
				case 2: {
					GpuMat d_buf_0, d_buf_1;
					d_frame_0.convertTo(d_buf_0, CV_32F, 1.0 / 255.0);
					d_frame_1.convertTo(d_buf_1, CV_32F, 1.0 / 255.0);
					alg_brox(d_buf_0, d_buf_1, d_flow_x, d_flow_y);
					break;
				}
				default:
					LOG(ERROR)<<"Unknown optical method: "<<type;
			}



			//get back flow map
			d_flow_x.download(flow_x);
			d_flow_y.download(flow_y);

			// warp to reduce holistic motion
			detector_surf.detect(capture_gray, kpts_surf, human_mask);
			extractor_surf.compute(capture_gray, kpts_surf, desc_surf);
			ComputeMatch(prev_kpts_surf, kpts_surf, prev_desc_surf, desc_surf, prev_pts_surf, pts_surf);
			MatchFromFlow_copy(capture_gray, flow_x, flow_y, prev_pts_flow, pts_flow, human_mask);
			MergeMatch(prev_pts_flow, pts_flow, prev_pts_surf, pts_surf, prev_pts_all, pts_all);
			Mat H = Mat::eye(3, 3, CV_64FC1);
			if(pts_all.size() > 50) {
				std::vector<unsigned char> match_mask;
				Mat temp = findHomography(prev_pts_all, pts_all, RANSAC, 1, match_mask);
				if(countNonZero(Mat(match_mask)) > 25)
					H = temp;
			}

			Mat H_inv = H.inv();
			Mat gray_warp = Mat::zeros(capture_gray.size(), CV_8UC1);
			MyWarpPerspective(prev_gray, capture_gray, gray_warp, H_inv);

			// re-extract flow on warped images
			d_frame_0.upload(prev_gray);
			d_frame_1.upload(gray_warp);

			switch(type){
				case 0: {
					alg_farn(d_frame_0, d_frame_1, d_flow_x, d_flow_y);
					break;
				}
				case 1: {
					alg_tvl1(d_frame_0, d_frame_1, d_flow_x, d_flow_y);
					break;
				}
				case 2: {
					GpuMat d_buf_0, d_buf_1;
					d_frame_0.convertTo(d_buf_0, CV_32F, 1.0 / 255.0);
					d_frame_1.convertTo(d_buf_1, CV_32F, 1.0 / 255.0);
					alg_brox(d_buf_0, d_buf_1, d_flow_x, d_flow_y);
					break;
				}
				default:
					LOG(ERROR)<<"Unknown optical method: "<<type;
			}



			//get back flow map
			d_flow_x.download(flow_x);
			d_flow_y.download(flow_y);

			vector<uchar> str_x, str_y;
			encodeFlowMap(flow_x, flow_y, str_x, str_y, bound);

			output_x.push_back(str_x);
			output_y.push_back(str_y);

			std::swap(prev_gray, capture_gray);
			std::swap(prev_image, capture_image);


			//get next frame
			bool hasnext = true;
			for(int s = 0; s < step; ++s){
				video_stream >> capture_frame;
				cnt ++;
				hasnext = !capture_frame.empty();
				// read frames until end
			}
			if (!hasnext){
				return;
			}
		}


	}
}