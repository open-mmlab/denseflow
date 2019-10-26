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
    file_id = H5Fcreate(hdf5_savepath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
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
            // std::cout << root_dir + "/" + curr_line << std::endl;
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
            initialized = true;
            cnt++;
        } else {
            std::clock_t begin = std::clock();
            if (!do_resize)
                capture_frame.copyTo(capture_image);
            else
                cv::resize(capture_frame, capture_image, new_size);

            cvtColor(capture_image, capture_gray, cv::COLOR_BGR2GRAY);
            d_frame_0.upload(prev_gray);
            d_frame_1.upload(capture_gray);
            std::clock_t upload = std::clock();
            t_upload += double(upload - begin) / CLOCKS_PER_SEC;

            switch (type) {
            case 0: {
                alg_farn->calc(d_frame_0, d_frame_1, d_flow);
                break;
            }
            case 1: {
                // if (cnt > 1)
                //     alg_tvl1->setUseInitialFlow(true);
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
            std::clock_t tvl1 = std::clock();
            t_flow += double(tvl1 - upload) / CLOCKS_PER_SEC;

            GpuMat planes[2];
            cuda::split(d_flow, planes);

            // get back flow map
            Mat flow_x(planes[0]);
            Mat flow_y(planes[1]);

            /* for correction debug
                        if (cnt == 60) {
                            float* flow_ptr = flow_y.ptr<float>();
                            for (int i=0; i<5; i++) {
                                for (int j=10; j<15; j++) {
                                    std::cout << flow_ptr[i*flow_y.cols+j] << ", ";
                                }
                                std::cout << std::endl;
                            }
                        }
            */
            // save as .h5
            // std::cout << "file_id " << file_id << ", curr_line " << curr_line << (", /" + curr_line).c_str()
            //           << std::endl;
            std::clock_t beforeh5 = std::clock();
            t_splitflow += double(beforeh5 - tvl1) / CLOCKS_PER_SEC;
            frame_id = H5Gcreate(file_id, ("/" + curr_line).c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            string flow_x_dataset = "/" + curr_line + "/flow_x";
            string flow_y_dataset = "/" + curr_line + "/flow_y";
            hdf5_save_nd_dataset(frame_id, flow_x_dataset, flow_x);
            hdf5_save_nd_dataset(frame_id, flow_y_dataset, flow_y);
            status = H5Gclose(frame_id);
            std::clock_t afterh5 = std::clock();
            t_h5 += double(afterh5 - beforeh5) / CLOCKS_PER_SEC;

            // // save as images
            // std::vector<uchar> str_x, str_y, str_img;
            // encodeFlowMap(flow_x, flow_y, str_x, str_y, bound);
            // imencode(".jpg", capture_image, str_img);
            // output_x.push_back(str_x);
            // output_y.push_back(str_y);
            // output_img.push_back(str_img);
            // names.push_back(curr_line);

            if (cnt % save_duration == 0) {
                std::clock_t beforewrite = std::clock();

                // 关闭文件对象
                status = H5Fclose(file_id);
                if (status < 0) {
                    throw "Failed to save hdf5 file: " + hdf5_savepath;
                }
                // 重新打开文件对象
                file_id = H5Fopen((hdf5_savepath).c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
                // 保存图片
                // writeImagesV2(output_img, output_x, output_y, names, output_root_dir);
                // output_x.clear();
                // output_y.clear();
                // output_img.clear();
                // names.clear();
                std::clock_t afterwrite = std::clock();
                t_writeh5 += double(afterh5 - beforeh5) / CLOCKS_PER_SEC;

                std::cout << "Processed " << cnt << " frames." << std::endl;
                std::cout << "upload:" << t_upload / cnt << ", flow:" << t_flow / cnt
                          << ", splitflow:" << t_splitflow / cnt << ", h5:" << t_h5 / cnt
                          << ", writeh5:" << t_writeh5 / cnt << ", fetch:" << t_fetch / cnt << ", swp:" << t_swp / cnt
                          << std::endl;
            }

            std::clock_t beforefetch = std::clock();
            // prefetch while gpu is working
            bool hasnext = (bool)getline(ifs, curr_line);
            if (hasnext) {
                // std::cout << root_dir+"/"+curr_line << std::endl;
                capture_frame = cv::imread(root_dir + "/" + curr_line, IMREAD_COLOR); // video_stream >> capture_frame;
                cnt++;
            }

            if (!hasnext) {
                // 关闭文件对象
                status = H5Fclose(file_id);
                if (status < 0) {
                    throw "Failed to save hdf5 file: " + hdf5_savepath;
                }
                // 保存图片
                // writeImagesV2(output_img, output_x, output_y, names, output_root_dir);
                std::cout << "Processed " << cnt << " frames." << std::endl;
                return;
            }
            std::clock_t afterfetch = std::clock();
            t_fetch += double(afterfetch - beforefetch) / CLOCKS_PER_SEC;

            std::swap(prev_gray, capture_gray);
            std::swap(prev_image, capture_image);
            std::clock_t swap = std::clock();
            t_swp += double(afterfetch - beforefetch) / CLOCKS_PER_SEC;
        }
    }
}
