#include "dense_flow.h"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudacodec.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "utils.h"
#include <clue/thread_pool.hpp>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>
using namespace cv::cuda;

void calcDenseNvFlowVideoGPU(string video_path, string output_dir, string algorithm, int step, int bound, int new_width,
                             int new_height, int new_short, int dev_id, bool verbose) {

    // read all frames into cpu
    vector<string> vid_splits;
    SplitString(video_path, vid_splits, "/");
    string vid_name = vid_splits[vid_splits.size() - 1];

    double before_read = CurrentSeconds();
    VideoCapture video_stream(video_path);
    CHECK(video_stream.isOpened()) << "Cannot open video_path stream " << video_path;
    int width = video_stream.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = video_stream.get(cv::CAP_PROP_FRAME_HEIGHT);
    Size size(width, height);

    // check resize
    bool do_resize = true;
    if (new_width > 0 && new_height > 0) {
        size.width = new_width;
        size.height = new_height;
    } else if (new_width > 0 && new_height == 0) {
        size.width = new_width;
        size.height = (int)round(height * 1.0 / width * new_width);
    } else if (new_width == 0 && new_height > 0) {
        size.width = (int)round(width * 1.0 / height * new_height);
        size.height = new_height;
    } else if (new_short > 0 && min(width, height) > new_short) {
        if (width < height) {
            size.width = new_short;
            size.height = (int)round(height * 1.0 / width * new_short);
        } else {
            size.width = (int)round(width * 1.0 / height * new_short);
            size.height = new_short;
        }
    } else {
        do_resize = false;
    }

    // extract frames only
    if (step == 0) {
        vector<vector<uchar>> output_img;
        Mat capture_frame;
        while (true) {
            vector<uchar> str_img;
            video_stream >> capture_frame;
            if (capture_frame.empty())
                break;
            if (do_resize) {
                Mat resized_frame;
                resized_frame.create(size, CV_8UC3);
                cv::resize(capture_frame, resized_frame, size);
                imencode(".jpg", resized_frame, str_img);
            } else {
                imencode(".jpg", capture_frame, str_img);
            }
            output_img.push_back(str_img);
        }
        video_stream.release();
        int N = output_img.size();
        double end_read = CurrentSeconds();
        if (verbose)
            std::cout << N << " frames decoded into cpu, using " << (end_read - before_read) << "s" << std::endl;
        double before_write = CurrentSeconds();
        writeImages(output_img, output_dir + "/img");
        double end_write = CurrentSeconds();
        if (verbose)
            std::cout << N << " frames written to disk, using " << (end_read - before_read) << "s" << std::endl;

        std::cout << vid_name << " has " << N << " frames extracted in " << (end_write - before_read) << "s, "
                  << N / (end_write - before_read) << "fps" << std::endl;
        return;
    }

    // extract gray frames for flow
    vector<Mat> frames_gray;
    Mat capture_frame;
    while (true) {
        video_stream >> capture_frame;
        if (capture_frame.empty())
            break;
        Mat frame_gray;
        cvtColor(capture_frame, frame_gray, COLOR_BGR2GRAY);
        if (do_resize) {
            Mat resized_frame_gray;
            resized_frame_gray.create(size, CV_8UC1);
            cv::resize(frame_gray, resized_frame_gray, size);
            frames_gray.push_back(resized_frame_gray);
        } else {
            frames_gray.push_back(frame_gray);
        }
    }
    video_stream.release();
    int N = frames_gray.size();
    double end_read = CurrentSeconds();
    if (verbose)
        std::cout << N << " frames decoded into cpu, using " << (end_read - before_read) << "s" << std::endl;

    // optflow
    double before_flow = CurrentSeconds();

    Ptr<cuda::FarnebackOpticalFlow> alg_farn = cuda::FarnebackOpticalFlow::create();
    Ptr<cuda::OpticalFlowDual_TVL1> alg_tvl1 = cuda::OpticalFlowDual_TVL1::create();
    Ptr<cuda::BroxOpticalFlow> alg_brox = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);
    Ptr<NvidiaOpticalFlow_1_0> alg_nv = NvidiaOpticalFlow_1_0::create(
        size.width, size.height, NvidiaOpticalFlow_1_0::NVIDIA_OF_PERF_LEVEL::NV_OF_PERF_LEVEL_SLOW, true, false, false,
        dev_id);
    int M = N - abs(step);
    if (M <= 0)
        return;

    vector<Mat> flows(M);
    for (size_t i = 0; i < M; ++i) {
        Mat flow;
        int a = step > 0 ? i : i - step;
        int b = step > 0 ? i + step : i;

        if (algorithm == "nv") {
            alg_nv->calc(frames_gray[a], frames_gray[b], flow);
            alg_nv->upSampler(flow, size.width, size.height, alg_nv->getGridSize(), flows[i]);
        } else {
            GpuMat gray_a_gpu, gray_b_gpu, flow_gpu;
            gray_a_gpu.upload(frames_gray[a]);
            gray_b_gpu.upload(frames_gray[b]);
            if (algorithm == "tvl1") {
                alg_tvl1->calc(gray_a_gpu, gray_b_gpu, flow_gpu);
            } else if (algorithm == "farn") {
                alg_farn->calc(gray_a_gpu, gray_b_gpu, flow_gpu);
            } else if (algorithm == "brox") {
                GpuMat d_buf_0, d_buf_1;
                gray_a_gpu.convertTo(d_buf_0, CV_32F, 1.0 / 255.0);
                gray_b_gpu.convertTo(d_buf_1, CV_32F, 1.0 / 255.0);
                alg_brox->calc(d_buf_0, d_buf_1, flow_gpu);
            } else {
                LOG(ERROR) << "Unknown optical algorithm " << algorithm;
                return;
            }
            flow_gpu.download(flows[i]);
        }
    }
    double end_flow = CurrentSeconds();
    if (verbose)
        std::cout << M << " flows computed, using " << (end_flow - before_flow) << "s" << std::endl;

    // encode
    double before_encode = CurrentSeconds();
    vector<vector<uchar>> output_x, output_y;
    for (int i = 0; i < M; ++i) {
        Mat planes[2];
        split(flows[i], planes);
        Mat flow_x(planes[0]);
        Mat flow_y(planes[1]);
        vector<uchar> str_x, str_y;
        encodeFlowMap(flow_x, flow_y, str_x, str_y, bound);
        output_x.push_back(str_x);
        output_y.push_back(str_y);
    }
    double end_encode = CurrentSeconds();
    if (verbose)
        std::cout << M << " flows encodeed to img, using " << (end_encode - before_encode) << "s" << std::endl;

    double before_write = CurrentSeconds();
    writeImages(output_x, output_dir + "/flow_x");
    writeImages(output_y, output_dir + "/flow_y");
    double end_write = CurrentSeconds();
    if (verbose)
        std::cout << M << " flows written to disk, using " << (end_write - before_write) << "s" << std::endl;

    std::cout << vid_name << " has " << M << " flows finished in " << (end_write - before_read) << "s, "
              << M / (end_write - before_read) << "fps" << std::endl;
}
