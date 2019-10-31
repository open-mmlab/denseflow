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

void calcDenseNvFlowVideoGPU(string file_name, string video, string output_root_dir, int bound, int type, int dev_id,
                             int new_width, int new_height, bool save_img, bool save_jpg, bool save_h5, bool save_zip) {
    if (type != 1) {
        LOG(ERROR) << "not implemented: " << type;
    }

    // read all pairs
    std::ifstream ifs(file_name);
    if (!ifs) {
        std::cout << "Cannot open file " << file_name << std::endl;
        return;
    }
    string line;
    vector<size_t> idAs, idBs;
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
    vector<Mat> frames;
    while (true) {
        Mat capture_frame;
        video_stream >> capture_frame;
        if (capture_frame.empty())
            break;
        frames.push_back(capture_frame);
    }
    video_stream.release();
    int N = frames.size();
    double end_read = CurrentSeconds();
    std::cout << N << " frames decoded into cpu, using " << (end_read - before_read) << "s" << std::endl;

    // optflow
    double before_flow = CurrentSeconds();
    Ptr<NvidiaOpticalFlow_1_0> nvof = NvidiaOpticalFlow_1_0::create(
        size.width, size.height, NvidiaOpticalFlow_1_0::NVIDIA_OF_PERF_LEVEL::NV_OF_PERF_LEVEL_SLOW, true, false, false,
        dev_id);
    vector<Mat> flows(M);
    for (size_t i = 0; i < M; ++i) {
        Mat flow;
        nvof->calc(frames[idAs[i]], frames[idBs[i]], flow);
        nvof->upSampler(flows[i], size.width, size.height, nvof->getGridSize(), flows[i]);
    }
    double end_flow = CurrentSeconds();
    std::cout << M << " flows computed, using " << (end_flow - before_flow) << "s" << std::endl;

    // download
    vector<vector<uchar>> output_x, output_y;
    double before_download = CurrentSeconds();
    for (int i = 0; i < M; ++i) {
        Mat planes[2];
        split(flows[i], planes);

        // get back flow map
        Mat flow_x(planes[0]);
        Mat flow_y(planes[1]);

        std::vector<uchar> str_x, str_y;
        encodeFlowMap(flow_x, flow_y, str_x, str_y, bound);
        output_x.push_back(str_x);
        output_y.push_back(str_y);
    }
    double end_download = CurrentSeconds();
    std::cout << M << " flows downloaded to cpu, using " << (end_download - before_download) << "s" << std::endl;

    double before_write = CurrentSeconds();
    writeImages(output_x, output_root_dir + "/flow_x");
    writeImages(output_y, output_root_dir + "/flow_y");
    double end_write = CurrentSeconds();
    std::cout << M << " flows wrote to disk, using " << (end_write - before_write) << "s" << std::endl;
}
