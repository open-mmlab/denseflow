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
using namespace cv;
using namespace cv::gpu;


static void MyWarpPerspective(Mat& prev_src, Mat& src, Mat& dst, Mat& M0, int flags = INTER_LINEAR,
	            			 int borderType = BORDER_CONSTANT, const Scalar& borderValue = Scalar())
{
	int width = src.cols;
	int height = src.rows;
	dst.create( height, width, CV_8UC1 );

	Mat mask = Mat::zeros(height, width, CV_8UC1);
	const int margin = 5;

    const int BLOCK_SZ = 32;
    short XY[BLOCK_SZ*BLOCK_SZ*2], A[BLOCK_SZ*BLOCK_SZ];

    int interpolation = flags & INTER_MAX;
    if( interpolation == INTER_AREA )
        interpolation = INTER_LINEAR;

    double M[9];
    Mat matM(3, 3, CV_64F, M);
    M0.convertTo(matM, matM.type());
    if( !(flags & WARP_INVERSE_MAP) )
         invert(matM, matM);

    int x, y, x1, y1;

    int bh0 = min(BLOCK_SZ/2, height);
    int bw0 = min(BLOCK_SZ*BLOCK_SZ/bh0, width);
    bh0 = min(BLOCK_SZ*BLOCK_SZ/bw0, height);

    for( y = 0; y < height; y += bh0 ) {
    for( x = 0; x < width; x += bw0 ) {
		int bw = min( bw0, width - x);
        int bh = min( bh0, height - y);

        Mat _XY(bh, bw, CV_16SC2, XY);
		Mat matA;
        Mat dpart(dst, Rect(x, y, bw, bh));

		for( y1 = 0; y1 < bh; y1++ ) {

			short* xy = XY + y1*bw*2;
            double X0 = M[0]*x + M[1]*(y + y1) + M[2];
            double Y0 = M[3]*x + M[4]*(y + y1) + M[5];
            double W0 = M[6]*x + M[7]*(y + y1) + M[8];
            short* alpha = A + y1*bw;

            for( x1 = 0; x1 < bw; x1++ ) {

                double W = W0 + M[6]*x1;
                W = W ? INTER_TAB_SIZE/W : 0;
                double fX = max((double)INT_MIN, min((double)INT_MAX, (X0 + M[0]*x1)*W));
                double fY = max((double)INT_MIN, min((double)INT_MAX, (Y0 + M[3]*x1)*W));

				double _X = fX/double(INTER_TAB_SIZE);
				double _Y = fY/double(INTER_TAB_SIZE);

				if( _X > margin && _X < width-1-margin && _Y > margin && _Y < height-1-margin )
					mask.at<uchar>(y+y1, x+x1) = 1;

                int X = saturate_cast<int>(fX);
                int Y = saturate_cast<int>(fY);

                xy[x1*2] = saturate_cast<short>(X >> INTER_BITS);
                xy[x1*2+1] = saturate_cast<short>(Y >> INTER_BITS);
                alpha[x1] = (short)((Y & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE + (X & (INTER_TAB_SIZE-1)));
            }
        }

        Mat _matA(bh, bw, CV_16U, A);
        remap( src, dpart, _XY, _matA, interpolation, borderType, borderValue );
    }
    }

	for( y = 0; y < height; y++ ) {
		const uchar* m = mask.ptr<uchar>(y);
		const uchar* s = prev_src.ptr<uchar>(y);
		uchar* d = dst.ptr<uchar>(y);
		for( x = 0; x < width; x++ ) {
			if(m[x] == 0)
				d[x] = s[x];
		}
	}
}

void ComputeMatch(const std::vector<KeyPoint>& prev_kpts, const std::vector<KeyPoint>& kpts,
				  const Mat& prev_desc, const Mat& desc, std::vector<Point2f>& prev_pts, std::vector<Point2f>& pts)
{
	prev_pts.clear();
	pts.clear();

	if(prev_kpts.size() == 0 || kpts.size() == 0)
		return;

	Mat mask = windowedMatchingMask(kpts, prev_kpts, 25, 25);

	BFMatcher desc_matcher(NORM_L2);
	std::vector<DMatch> matches;

	desc_matcher.match(desc, prev_desc, matches, mask);

	prev_pts.reserve(matches.size());
	pts.reserve(matches.size());

	for(size_t i = 0; i < matches.size(); i++) {
		const DMatch& dmatch = matches[i];
		// get the point pairs that are successfully matched
		prev_pts.push_back(prev_kpts[dmatch.trainIdx].pt);
		pts.push_back(kpts[dmatch.queryIdx].pt);
	}

	return;
}

void MergeMatch(const std::vector<Point2f>& prev_pts1, const std::vector<Point2f>& pts1,
				const std::vector<Point2f>& prev_pts2, const std::vector<Point2f>& pts2,
				std::vector<Point2f>& prev_pts_all, std::vector<Point2f>& pts_all)
{
	prev_pts_all.clear();
	prev_pts_all.reserve(prev_pts1.size() + prev_pts2.size());

	pts_all.clear();
	pts_all.reserve(pts1.size() + pts2.size());

	for(size_t i = 0; i < prev_pts1.size(); i++) {
		prev_pts_all.push_back(prev_pts1[i]);
		pts_all.push_back(pts1[i]);
	}

	for(size_t i = 0; i < prev_pts2.size(); i++) {
		prev_pts_all.push_back(prev_pts2[i]);
		pts_all.push_back(pts2[i]);
	}

	return;
}

void MatchFromFlow(const Mat& prev_grey, const Mat& flow, std::vector<Point2f>& prev_pts, std::vector<Point2f>& pts, const Mat& mask)
{
	int width = prev_grey.cols;
	int height = prev_grey.rows;
	prev_pts.clear();
	pts.clear();

	const int MAX_COUNT = 1000;
	goodFeaturesToTrack(prev_grey, prev_pts, MAX_COUNT, 0.001, 3, mask);

	if(prev_pts.size() == 0)
		return;

	for(int i = 0; i < prev_pts.size(); i++) {
		int x = std::min<int>(std::max<int>(cvRound(prev_pts[i].x), 0), width-1);
		int y = std::min<int>(std::max<int>(cvRound(prev_pts[i].y), 0), height-1);

		const float* f = flow.ptr<float>(y);
		pts.push_back(Point2f(x+f[2*x], y+f[2*x+1]));
	}
}

void MatchFromFlow_copy(const Mat& prev_grey, const Mat& flow_x, const Mat& flow_y, std::vector<Point2f>& prev_pts, std::vector<Point2f>& pts, const Mat& mask)
{
	int width = prev_grey.cols;
	int height = prev_grey.rows;
	prev_pts.clear();
	pts.clear();

	const int MAX_COUNT = 1000;
	goodFeaturesToTrack(prev_grey, prev_pts, MAX_COUNT, 0.001, 3, mask);

	if(prev_pts.size() == 0)
		return;

	for(int i = 0; i < prev_pts.size(); i++) {
		int x = std::min<int>(std::max<int>(cvRound(prev_pts[i].x), 0), width-1);
		int y = std::min<int>(std::max<int>(cvRound(prev_pts[i].y), 0), height-1);

		const float* f_x = flow_x.ptr<float>(y);
		const float* f_y = flow_y.ptr<float>(y);
		pts.push_back(Point2f(x+f_x[x], y+f_y[y]));
	}
}

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