#include "common.h"

int main(int argc, char** argv)
{
	// IO operation

	const char* keys =
		{
			"{ f  | vidFile      | ex2.avi | filename of video }"
			"{ x  | xFlowFile    | flow_x | filename of flow x component }"
			"{ y  | yFlowFile    | flow_y | filename of flow x component }"
			"{ i  | imgFile      | flow_i | filename of flow image}"
			"{ b  | bound | 15 | specify the maximum of optical flow}"
		};

	CommandLineParser cmd(argc, argv, keys);
	string vidFile = cmd.get<string>("vidFile");
	string xFlowFile = cmd.get<string>("xFlowFile");
	string yFlowFile = cmd.get<string>("yFlowFile");
	string imgFile = cmd.get<string>("imgFile");
	int bound = cmd.get<int>("bound");


	VideoCapture capture(vidFile);
	if(!capture.isOpened()) {
		printf("Could not initialize capturing..\n");
		return -1;
	}

	int frame_num = 0;
	Mat image, prev_image, prev_grey, grey, frame, flow, cflow;

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
			continue;
		}

		frame.copyTo(image);
		cvtColor(image, grey, CV_BGR2GRAY);

		// calcOpticalFlowFarneback(prev_grey,grey,flow,0.5, 3, 15, 3, 5, 1.2, 0 );
    calcOpticalFlowFarneback(prev_grey, grey, flow, 0.702, 5, 10, 2, 7, 1.5, cv::OPTFLOW_FARNEBACK_GAUSSIAN );
		
		// prev_image.copyTo(cflow);
		// drawOptFlowMap(flow, cflow, 12, 1.5, Scalar(0, 255, 0));

		Mat flows[2];
		split(flow,flows);
		Mat imgX(flows[0].size(),CV_8UC1);
		Mat imgY(flows[0].size(),CV_8UC1);
		convertFlowToImage(flows[0],flows[1], imgX, imgY, -bound, bound);
		char tmp[20];
		sprintf(tmp,"_%04d.jpg",int(frame_num));
		imwrite(xFlowFile + tmp,imgX);
		imwrite(yFlowFile + tmp,imgY);
		imwrite(imgFile + tmp, image);

		std::swap(prev_grey, grey);
		std::swap(prev_image, image);
		frame_num = frame_num + 1;
	}
	return 0;
}
