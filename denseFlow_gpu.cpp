#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"

#include <stdio.h>
#include <iostream>
using namespace cv;
using namespace cv::gpu;

static void convertFlowToImage(const Mat &flow_x, const Mat &flow_y, Mat &img_x, Mat &img_y,
       double lowerBound, double higherBound) {
	#define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255*((v) - (L))/((H)-(L))))
	for (int i = 0; i < flow_x.rows; ++i) {
		for (int j = 0; j < flow_y.cols; ++j) {
			float x = flow_x.at<float>(i,j);
			float y = flow_y.at<float>(i,j);
			img_x.at<uchar>(i,j) = CAST(x, lowerBound, higherBound);
			img_y.at<uchar>(i,j) = CAST(y, lowerBound, higherBound);
		}
	}
	#undef CAST
}

static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,double, const Scalar& color){
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            circle(cflowmap, Point(x,y), 2, color, -1);
        }
}

int main(int argc, char** argv){
	// IO operation
	const char* keys =
		{
			"{ f  | vidFile      | ex2.avi | filename of video }"
			"{ x  | xFlowFile    | flow_x | filename of flow x component }"
			"{ y  | yFlowFile    | flow_y | filename of flow x component }"
			"{ i  | imgFile      | flow_i | filename of flow image}"
			"{ b  | bound | 15 | specify the maximum of optical flow}"
			"{ t  | type | 0 | specify the optical flow algorithm }"
			"{ d  | device_id    | 0  | set gpu id}"
			"{ s  | step  | 1 | specify the step for frame sampling}"
		};

	CommandLineParser cmd(argc, argv, keys);
	string vidFile = cmd.get<string>("vidFile");
	string xFlowFile = cmd.get<string>("xFlowFile");
	string yFlowFile = cmd.get<string>("yFlowFile");
	string imgFile = cmd.get<string>("imgFile");
	int bound = cmd.get<int>("bound");
    int type  = cmd.get<int>("type");
    int device_id = cmd.get<int>("device_id");
    int step = cmd.get<int>("step");

	VideoCapture capture(vidFile);
	if(!capture.isOpened()) {
		printf("Could not initialize capturing..\n");
		return -1;
	}

	int frame_num = 0;
	Mat image, prev_image, prev_grey, grey, frame, flow_x, flow_y;
	GpuMat frame_0, frame_1, flow_u, flow_v;

	setDevice(device_id);
	FarnebackOpticalFlow alg_farn;
	OpticalFlowDual_TVL1_GPU alg_tvl1;
	BroxOpticalFlow alg_brox(0.197f, 50.0f, 0.8f, 10, 77, 10);

	while(true) {
		capture >> frame;
		if(frame.empty())
			break;
		if(frame_num == 0) {
			image.create(frame.size(), CV_8UC3);
			grey.create(frame.size(), CV_8UC1);
			prev_image.create(frame.size(), CV_8UC3);
			prev_grey.create(frame.size(), CV_8UC1);

			frame.copyTo(prev_image);
			cvtColor(prev_image, prev_grey, CV_BGR2GRAY);

			frame_num++;

			int step_t = step;
			while (step_t > 1){
				capture >> frame;
				step_t--;
			}
			continue;
		}

		frame.copyTo(image);
		cvtColor(image, grey, CV_BGR2GRAY);


		frame_0.upload(prev_grey);
		frame_1.upload(grey);


        // GPU optical flow
		switch(type){
		case 0:
			alg_farn(frame_0,frame_1,flow_u,flow_v);
			break;
		case 1:
			alg_tvl1(frame_0,frame_1,flow_u,flow_v);
			break;
		case 2:
			GpuMat d_frame0f, d_frame1f;
	        frame_0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
	        frame_1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);
			alg_brox(d_frame0f, d_frame1f, flow_u,flow_v);
			break;
		}

		flow_u.download(flow_x);
		flow_v.download(flow_y);

		// Output optical flow
		Mat imgX(flow_x.size(),CV_8UC1);
		Mat imgY(flow_y.size(),CV_8UC1);
		convertFlowToImage(flow_x,flow_y, imgX, imgY, -bound, bound);
		char tmp[20];
		sprintf(tmp,"_%04d.jpg",int(frame_num));

		Mat imgX_, imgY_, image_;
		resize(imgX,imgX_,cv::Size(340,256));
		resize(imgY,imgY_,cv::Size(340,256));
		resize(image,image_,cv::Size(340,256));

		imwrite(xFlowFile + tmp,imgX_);
		imwrite(yFlowFile + tmp,imgY_);
		imwrite(imgFile + tmp, image_);

		std::swap(prev_grey, grey);
		std::swap(prev_image, image);
		frame_num = frame_num + 1;

		int step_t = step;
		while (step_t > 1){
			capture >> frame;
			step_t--;
		}
	}
	return 0;
}
