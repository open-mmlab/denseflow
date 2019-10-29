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

void calcDenseFlowFramesGPU(string file_name, string root_dir, string output_root_dir, int bound, int type, int dev_id,
                            int new_width, int new_height) {
    std::ifstream ifs(file_name);
    if (!ifs) {
        std::cout << "Cannot open file_name \"" << file_name << "\" for optical flow extraction.";
        return;
    }

    bool save_h5 = false;
    bool save_img = true;
    bool save_flow = true;

    setDevice(dev_id);
    Mat capture_frame, capture_image, prev_image, capture_gray, prev_gray;
    Mat flow_x, flow_y;
    Size new_size(new_width, new_height);

    GpuMat d_frame_0, d_frame_1;
    GpuMat d_flow;

    cv::Ptr<cuda::FarnebackOpticalFlow> alg_farn = cuda::FarnebackOpticalFlow::create();
    cv::Ptr<cuda::OpticalFlowDual_TVL1> alg_tvl1 = cuda::OpticalFlowDual_TVL1::create();
    cv::Ptr<cuda::BroxOpticalFlow> alg_brox = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);

    bool do_resize = (new_height > 0) && (new_width > 0);

    bool initialized = false;
    int cnt = 0;
    string curr_line;

    vector<vector<uchar>> output_x, output_y, output_img;
    vector<string> names;
    hid_t file_id, frame_id; // hdf5
    herr_t status;
    const int save_duration = 128;
    vector<string> vid_splits;
    SplitString(root_dir, vid_splits, "/");
    string vid_name = vid_splits[vid_splits.size() - 1];
    string hdf5_savepath = output_root_dir + "/" + vid_name + ".h5";
    if (save_h5) {
        file_id = H5Fcreate(hdf5_savepath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    }
    double t_upload = 0;
    double t_flow = 0;
    double t_splitflow = 0;
    double t_h5 = 0;
    double t_writeh5 = 0;
    double t_fetch = 0;
    double t_swp = 0;

    while (true) {

        // build mats for the first frame
        if (!initialized) {
            getline(ifs, curr_line);
            capture_frame = cv::imread(root_dir + "/" + curr_line, IMREAD_COLOR); // video_stream >> capture_frame;
            if (capture_frame.empty())
                return; // read frames until end

            if (!do_resize) {
                initializeMats(capture_frame, capture_image, capture_gray, prev_image, prev_gray);
                capture_frame.copyTo(prev_image);
            } else {
                capture_image.create(new_size, CV_8UC3);
                capture_gray.create(new_size, CV_8UC1);
                prev_image.create(new_size, CV_8UC3);
                prev_gray.create(new_size, CV_8UC1);
                cv::resize(capture_frame, prev_image, new_size);
            }
            cvtColor(prev_image, prev_gray, cv::COLOR_BGR2GRAY);
            d_frame_1.upload(prev_gray);
            initialized = true;
            cnt++;
        } else {
            double begin = CurrentSeconds();
            if (!do_resize)
                capture_frame.copyTo(capture_image);
            else
                cv::resize(capture_frame, capture_image, new_size);

            cvtColor(capture_image, capture_gray, cv::COLOR_BGR2GRAY);
            d_frame_0.upload(capture_gray);
            cv::cuda::swap(d_frame_0, d_frame_1);
            double upload = CurrentSeconds();
            t_upload += upload - begin;

            switch (type) {
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
                LOG(ERROR) << "Unknown optical method: " << type;
            }
            double tvl1 = CurrentSeconds();
            t_flow += tvl1 - upload;

            GpuMat planes[2];
            cuda::split(d_flow, planes);

            // get back flow map
            Mat flow_x(planes[0]);
            Mat flow_y(planes[1]);

            // save as .h5
            double beforeh5 = CurrentSeconds();
            t_splitflow += beforeh5 - tvl1;
            if (save_h5) {
                frame_id = H5Gcreate(file_id, ("/" + curr_line).c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                string flow_x_dataset = "/" + curr_line + "/flow_x";
                string flow_y_dataset = "/" + curr_line + "/flow_y";
                hdf5_save_nd_dataset(frame_id, flow_x_dataset, flow_x);
                hdf5_save_nd_dataset(frame_id, flow_y_dataset, flow_y);
                status = H5Gclose(frame_id);
            }
            double afterh5 = CurrentSeconds();
            t_h5 += afterh5 - beforeh5;

            if (save_img) {
                // save as images
                std::vector<uchar> str_x, str_y, str_img;
                encodeFlowMap(flow_x, flow_y, str_x, str_y, bound);
                imencode(".jpg", capture_image, str_img);
                output_x.push_back(str_x);
                output_y.push_back(str_y);
                output_img.push_back(str_img);
                names.push_back(curr_line);
            }

            if (cnt % save_duration == 0) {
                double beforewrite = CurrentSeconds();

                if (save_h5) {
                    // 关闭文件对象
                    status = H5Fclose(file_id);
                    if (status < 0) {
                        throw "Failed to save hdf5 file: " + hdf5_savepath;
                    }
                    // 重新打开文件对象
                    file_id = H5Fopen((hdf5_savepath).c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
                }
                if (save_img) {
                    // 保存图片
                    writeImagesV2(output_img, output_x, output_y, names, output_root_dir);
                    output_x.clear();
                    output_y.clear();
                    output_img.clear();
                    names.clear();
                }
                double afterwrite = CurrentSeconds();
                t_writeh5 += afterwrite - beforewrite;

                std::cout << "Processed " << cnt << " frames." << std::endl;
                std::cout << "upload:" << t_upload / cnt << ", flow:" << t_flow / cnt
                          << ", splitflow:" << t_splitflow / cnt << ", h5:" << t_h5 / cnt
                          << ", writeh5:" << t_writeh5 / cnt << ", fetch:" << t_fetch / cnt << ", swp:" << t_swp / cnt
                          << std::endl;
            }

            double beforefetch = CurrentSeconds();
            // prefetch while gpu is working
            bool hasnext = (bool)getline(ifs, curr_line);
            if (hasnext) {
                capture_frame = cv::imread(root_dir + "/" + curr_line, IMREAD_COLOR); // video_stream >> capture_frame;
                cnt++;
            }

            if (!hasnext) {
                // 关闭文件对象
                if (save_h5) {
                    status = H5Fclose(file_id);
                }
                if (status < 0) {
                    throw "Failed to save hdf5 file: " + hdf5_savepath;
                }
                // 保存图片
                if (save_img) {
                    writeImagesV2(output_img, output_x, output_y, names, output_root_dir);
                }
                std::cout << "Processed " << cnt << " frames." << std::endl;
                std::cout << "upload:" << t_upload / cnt << ", flow:" << t_flow / cnt
                          << ", splitflow:" << t_splitflow / cnt << ", h5:" << t_h5 / cnt
                          << ", writeh5:" << t_writeh5 / cnt << ", fetch:" << t_fetch / cnt << ", swp:" << t_swp / cnt
                          << std::endl;
                return;
            }
            double afterfetch = CurrentSeconds();
            t_fetch += afterfetch - beforefetch;
        }
    }
}
