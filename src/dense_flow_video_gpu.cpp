//
// Created by yjxiong on 11/18/15.
//
#include "dense_flow.h"
#include "hdf5_utils.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudacodec.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "utils.h"
#include <ctime>
#include <fstream>
#include <string>
#include <vector>
using namespace cv::cuda;

void calcDenseFlowVideoGPU(string file_name, string video, string output_root_dir, int bound, int type, int dev_id,
                           int new_width, int new_height, bool save_img, bool save_jpg, bool save_h5, bool save_zip) {
    // timers
    double t_upload = 0;
    double t_flow = 0;
    double t_splitflow = 0;
    double t_h5 = 0;
    double t_writeh5 = 0;
    double t_fetch = 0;
    double t_swp = 0;
    cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator(cv::cuda::HostMem::AllocType::PAGE_LOCKED));

    // read all pairs
    std::ifstream ifs(file_name);
    if (!ifs) {
        std::cout << "Cannot open file " << file_name << std::endl;
        return;
    }
    string line;
    vector<uint> idAs, idBs;
    while (getline(ifs, line)) {
        std::size_t delimpos = line.find('\t');
        if (delimpos == string::npos)
            break;
        int a, b;
        idAs.push_back(std::stoi(line.substr(0, delimpos)));
        idBs.push_back(std::stoi(line.substr(delimpos + 1, line.length())));
    }
    ifs.close();
    size_t M = idAs.size();
    std::cout << M << " flows of " << video << " to compute." << std::endl;

    // read all frames into cpu
    double before_read = CurrentSeconds();
    VideoCapture video_stream(video);
    CHECK(video_stream.isOpened()) << "Cannot open video stream " << video;
    int width = video_stream.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = video_stream.get(cv::CAP_PROP_FRAME_HEIGHT);
    Size size(width, height);
    vector<Mat> frames_gray_cpu;
    Mat capture_frame_cpu;
    while (true) {
        video_stream >> capture_frame_cpu;
        if (capture_frame_cpu.empty())
            break;
        Mat frame_gray_cpu;
        cvtColor(capture_frame_cpu, frame_gray_cpu, COLOR_BGR2GRAY);
        frames_gray_cpu.push_back(frame_gray_cpu);
    }
    video_stream.release();
    int N = frames_gray_cpu.size();
    double end_read = CurrentSeconds();
    std::cout << N << " frames loaded into cpu, using " << (end_read - before_read) << "s" << std::endl;

    // upload all frames into gpu
    double before_upload = CurrentSeconds();
    setDevice(dev_id);
    size_t P = 4;
    vector<cv::cuda::Stream> streams(P);
    vector<GpuMat> frames_gray(N);
    for (int i = 0; i < N; ++i) {
        frames_gray[i].upload(frames_gray_cpu[i], streams[i % P]);
    }
    double end_upload = CurrentSeconds();
    std::cout << N << " frames uploaded into gpu, using " << (end_upload - before_upload) << "s" << std::endl;

    // optflow
    double before_flow = CurrentSeconds();
    cv::Ptr<cuda::OpticalFlowDual_TVL1> alg_tvl1 = cuda::OpticalFlowDual_TVL1::create();
    vector<GpuMat> flows(M);
    for (int i = 0; i < M; ++i) {
        switch (type) {
        case 0: {
            LOG(ERROR) << "not implemented: " << type;
        }
        case 1: {
            alg_tvl1->calc(frames_gray[idAs[i]], frames_gray[idBs[i]], flows[i], streams[i % P]);
            break;
        }
        case 2: {
            LOG(ERROR) << "not implemented: " << type;
        }
        default:
            LOG(ERROR) << "Unknown optical method: " << type;
        }
    }
    for (int i = 0; i < P; ++i) {
        streams[i].waitForCompletion();
    }
    double end_flow = CurrentSeconds();
    std::cout << M << " flows computed, using " << (end_flow - before_flow) << "s" << std::endl;

    // download
    vector<vector<uchar>> output_x, output_y;
    double before_download = CurrentSeconds();
    for (int i = 0; i < M; ++i) {
        GpuMat planes[2];
        cuda::split(flows[i], planes);

        // get back flow map
        Mat flow_x(planes[0]);
        Mat flow_y(planes[1]);

        std::vector<uchar> str_x, str_y;
        encodeFlowMap(flow_x, flow_y, str_x, str_y, bound);
        output_x.push_back(str_x);
        output_y.push_back(str_y);

        // frames_gray[i].download(planes[0], streams[i % P]);
    }
    double end_download = CurrentSeconds();
    std::cout << M << " flows downloaded to cpu, using " << (end_download - before_download) << "s" << std::endl;

    double before_write = CurrentSeconds();
    writeImages(output_x, output_root_dir + "/flow_x");
    writeImages(output_y, output_root_dir + "/flow_y");
    double end_write = CurrentSeconds();
    std::cout << M << " flows wrote to disk, using " << (end_write - before_write) << "s" << std::endl;

}
