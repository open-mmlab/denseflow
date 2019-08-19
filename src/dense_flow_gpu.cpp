//
// Created by yjxiong on 11/18/15.
//
#include "dense_flow.h"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudacodec.hpp"

using namespace cv::cuda;


void calcDenseFlowGPU(string file_name, int bound, int type, int step, int dev_id,
                      vector<vector<uchar> >& output_x,
                      vector<vector<uchar> >& output_y,
                      vector<vector<uchar> >& output_img,
                      int new_width, int new_height){
    VideoCapture video_stream(file_name);
    CHECK(video_stream.isOpened())<<"Cannot open video stream \""
                                  <<file_name
                                  <<"\" for optical flow extraction.";

    setDevice(dev_id);
    Mat capture_frame, capture_image, prev_image, capture_gray, prev_gray;
    Mat flow_x, flow_y;
    Size new_size(new_width, new_height);

    GpuMat d_frame_0, d_frame_1;
    GpuMat d_flow;

    cv::Ptr<cuda::FarnebackOpticalFlow> alg_farn = cuda::FarnebackOpticalFlow::create();
    cv::Ptr<cuda::OpticalFlowDual_TVL1> alg_tvl1 = cuda::OpticalFlowDual_TVL1::create();
    cv::Ptr<cuda::BroxOpticalFlow> alg_brox      = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);

    bool do_resize = (new_height > 0) && (new_width > 0);

    bool initialized = false;
    int cnt = 0;
    while(true){

        //build mats for the first frame
        if (!initialized){
           video_stream >> capture_frame;
           if (capture_frame.empty()) return; // read frames until end

            if (!do_resize){
                initializeMats(capture_frame, capture_image, capture_gray,
                           prev_image, prev_gray);
                capture_frame.copyTo(prev_image);
            }else{
                capture_image.create(new_size, CV_8UC3);
                capture_gray.create(new_size, CV_8UC1);
                prev_image.create(new_size, CV_8UC3);
                prev_gray.create(new_size, CV_8UC1);
                cv::resize(capture_frame, prev_image, new_size);
            }
            cvtColor(prev_image, prev_gray, COLOR_BGR2GRAY);
            initialized = true;
            for(int s = 0; s < step; ++s){
                video_stream >> capture_frame;
        cnt ++;
                if (capture_frame.empty()) return; // read frames until end
            }
        }else {
            if (!do_resize)
                capture_frame.copyTo(capture_image);
            else
                cv::resize(capture_frame, capture_image, new_size);

            cvtColor(capture_image, capture_gray, COLOR_BGR2GRAY);
            d_frame_0.upload(prev_gray);
            d_frame_1.upload(capture_gray);

            switch(type){
                case 0: {
                    alg_farn->calc(d_frame_0, d_frame_1, d_flow);
                    break;
                }
                case 1: {
                    alg_tvl1->calc(d_frame_0, d_frame_1, d_flow);
                    break;
                }
                case 2: {
                    GpuMat d_buf_0, d_buf_1;
                    d_frame_0.convertTo(d_buf_0, CV_32F, 1.0 / 255.0);
                    d_frame_1.convertTo(d_buf_1, CV_32F, 1.0 / 255.0);
                    alg_brox->calc(d_buf_0, d_buf_1, d_flow);
                    break;
                }
                default:
                    LOG(ERROR)<<"Unknown optical method: "<<type;
            }

            //prefetch while gpu is working
            bool hasnext = true;
            for(int s = 0; s < step; ++s){
                video_stream >> capture_frame;
		cnt ++;
                hasnext = !capture_frame.empty();
                // read frames until end
            }

            GpuMat planes[2];
            cuda::split(d_flow, planes);

            //get back flow map
            Mat flow_x(planes[0]);
            Mat flow_y(planes[1]);

            std::vector<uchar> str_x, str_y, str_img;
            encodeFlowMap(flow_x, flow_y, str_x, str_y, bound);
            imencode(".jpg", capture_image, str_img);

            output_x.push_back(str_x);
            output_y.push_back(str_y);
            output_img.push_back(str_img);

            std::swap(prev_gray, capture_gray);
            std::swap(prev_image, capture_image);

            if (!hasnext){
                return;
            }
        }


    }

}

/**
 * This function use pure GPU backend for video loading and optical flow calculation
 */
void calcDenseFlowPureGPU(std::string file_name, int bound, int type, int step, int dev_id,
                      std::vector<std::vector<uchar> >& output_x,
                      std::vector<std::vector<uchar> >& output_y,
                      std::vector<std::vector<uchar> >& output_img){

    setDevice(dev_id);
    cv::Ptr<cudacodec::VideoReader> video_stream = cudacodec::createVideoReader(file_name);
//    VideoCapture video_stream(file_name);
    //CHECK(video_stream->isOpened())<<"Cannot open video stream \""
    //                              <<file_name
    //                              <<"\" for optical flow extraction.";

    GpuMat capture_frame, capture_image, prev_image, capture_gray, prev_gray;
    Mat flow_x, flow_y, img;

    GpuMat d_flow;

    cv::Ptr<cuda::FarnebackOpticalFlow> alg_farn = cuda::FarnebackOpticalFlow::create();
    cv::Ptr<cuda::OpticalFlowDual_TVL1> alg_tvl1 = cuda::OpticalFlowDual_TVL1::create();
    cv::Ptr<cuda::BroxOpticalFlow> alg_brox      = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);

    bool initialized = false;
    while(true){

        //build mats for the first frame
        if (!initialized){
            bool success = video_stream->nextFrame(capture_frame);
            if (!success) break; // read frames until end
            capture_image.create(capture_frame.size(), CV_8UC3);
            capture_gray.create(capture_frame.size(), CV_8UC1);

            prev_image.create(capture_frame.size(), CV_8UC3);
            prev_gray.create(capture_frame.size(), CV_8UC1);

            capture_frame.copyTo(prev_image);
            cvtColor(prev_image, prev_gray, COLOR_BGR2GRAY);
            initialized = true;

            for (int s = 0; s < step; ++s){
                video_stream->nextFrame(capture_frame);
            }
        }else {
            capture_frame.copyTo(capture_image);
            cvtColor(capture_image, capture_gray, COLOR_BGR2GRAY);

            switch(type){
                case 0: {
                    alg_farn->calc(prev_gray, capture_gray, d_flow);
                    break;
                }
                case 1: {
                    alg_tvl1->calc(prev_gray, capture_gray, d_flow);
                    break;
                }
                case 2: {
                    GpuMat d_buf_0, d_buf_1;
                    prev_gray.convertTo(d_buf_0, CV_32F, 1.0 / 255.0);
                    capture_gray.convertTo(d_buf_1, CV_32F, 1.0 / 255.0);
                    alg_brox->calc(d_buf_0, d_buf_1, d_flow);
                    break;
                }
                default:
                    LOG(ERROR)<<"Unknown optical method: "<<type;
            }

            for (int s = 0; s < step; ++s){
                if (!video_stream->nextFrame(capture_frame)) break;
            }

            GpuMat planes[2];
            cuda::split(d_flow, planes);

            //get back flow map
            Mat flow_x(planes[0]);
            Mat flow_y(planes[1]);
            capture_image.download(img);

            std::vector<uchar> str_x, str_y, str_img;
            encodeFlowMap(flow_x, flow_y, str_x, str_y, bound);
            imencode(".jpg", img, str_img);

            output_x.push_back(str_x);
            output_y.push_back(str_y);
            output_img.push_back(str_img);

            std::swap(prev_gray, capture_gray);
            std::swap(prev_image, capture_image);
        }


    }

}
