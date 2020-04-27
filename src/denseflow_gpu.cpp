#include "dense_flow.h"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "utils.h"
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

bool DenseFlow::check_param() {
    for (int i = 0; i < video_paths.size(); i++) {
        if (!exists(video_paths[i])) {
            cout << video_paths[i] << " does not exist!" << endl;
            return false;
        }
        if (!is_directory(output_dirs[i])) {
            cout << output_dirs[i] << " is not a valid dir!" << endl;
            return false;
        }
    }
    if (algorithm != "nv" && algorithm != "tvl1" && algorithm != "farn" && algorithm != "brox") {
        cout << algorithm << " not supported!" << endl;
        return false;
    }
    if (bound <= 0) {
        cout << "bound should > 0!" << endl;
        return false;
    }
    if (new_height < 0 || new_width < 0 || new_short < 0) {
        cout << "height and width cannot < 0!" << endl;
        return false;
    }
    if (new_short > 0 && new_height + new_width != 0) {
        cout << "do not set height and width when set short!" << endl;
        return false;
    }
    if (save_type != "jpg" && save_type != "png" && save_type != "h5") {
        cout << "only jpg/png/h5 are supported (no " << save_type << ") for output" << endl;
        return false;
    }

    return true;
}

bool DenseFlow::get_new_size(const VideoCapture &video_stream, const vector<path> &frames_path, bool use_frames,
                             Size &new_size, int &frames_num) {
    int width, height;
    if (use_frames) {
        Mat src = imread(frames_path[0].string());
        width = src.size().width;
        height = src.size().height;
        frames_num = frames_path.size();
    } else {
        width = video_stream.get(cv::CAP_PROP_FRAME_WIDTH);
        height = video_stream.get(cv::CAP_PROP_FRAME_HEIGHT);
        frames_num = video_stream.get(cv::CAP_PROP_FRAME_COUNT);
    }
    // check resize
    bool do_resize = true;
    if (new_width > 0 && new_height > 0) {
        new_size.width = new_width;
        new_size.height = new_height;
    } else if (new_width > 0 && new_height == 0) {
        new_size.width = new_width;
        new_size.height = (int)round(height * 1.0 / width * new_width);
    } else if (new_width == 0 && new_height > 0) {
        new_size.width = (int)round(width * 1.0 / height * new_height);
        new_size.height = new_height;
    } else if (new_short > 0 && min(width, height) > new_short) {
        if (width < height) {
            new_size.width = new_short;
            new_size.height = (int)round(height * 1.0 / width * new_short);
        } else {
            new_size.width = (int)round(width * 1.0 / height * new_short);
            new_size.height = new_short;
        }
    } else {
        do_resize = false;
    }
    return do_resize;
}

int DenseFlow::extract_frames_video(VideoCapture &video_stream, vector<path> &frames_path, bool use_frames,
                                    bool do_resize, const Size &size, path output_dir, bool verbose) {
    int video_frame_idx = 0;
    while (true) {
        vector<Mat> frames_color;
        bool is_open = load_frames_batch(video_stream, frames_path, use_frames, frames_color, do_resize, size, false);
        vector<vector<uchar>> output_img;
        int N = frames_color.size();
        for (size_t i = 0; i < N; i++) {
            vector<uchar> str_img;
            imencode(".jpg", frames_color[i], str_img);
            output_img.push_back(str_img);
        }
        writeImages(output_img, (output_dir / "img").c_str(), video_frame_idx);
        video_frame_idx += N;
        if (!is_open) {
            break;
        }
        if (use_frames) {
            frames_path.erase(frames_path.begin(), frames_path.begin() + N);
        }
    }
    return video_frame_idx;
}

void DenseFlow::extract_frames_only(bool use_frames, bool verbose) {
    for (size_t i = 0; i < video_paths.size(); i++) {
        path video_path = video_paths[i];
        path output_dir = output_dirs[i];
        VideoCapture video_stream;
        vector<path> frames_path;
        if (use_frames) {
            directory_iterator end_itr;
            for (directory_iterator itr(video_path); itr != end_itr; ++itr) {
                if (!is_regular_file(itr->status()) || itr->path().extension() != ".jpg")
                    continue;
                frames_path.push_back(itr->path());
            }
            if (frames_path.size() == 0) {
                if (verbose)
                    cout << video_path << " is empty!" << endl;
                continue;
            }
            sort(frames_path.begin(), frames_path.end());
        } else {
            video_stream.open(video_path.c_str());
            if (!video_stream.isOpened())
                throw std::runtime_error("cannot open video_path stream:" + video_path.string());
        }

        Size size;
        int frames_num;
        bool do_resize = get_new_size(video_stream, frames_path, use_frames, size, frames_num);
        if (verbose)
            cout << video_path << ", frames ≈ " << frames_num << endl;
        frames_num = extract_frames_video(video_stream, frames_path, use_frames, do_resize, size, output_dir, verbose);
        total_frames += frames_num; // exact frame count
        if (!use_frames)
            video_stream.release();
        if (verbose)
            cout << "extracted frames of video " << video_path << ", " << frames_num << " frames" << endl;
    }
}

bool DenseFlow::load_frames_batch(VideoCapture &video_stream, const vector<path> &frames_path, bool use_frames,
                                  vector<Mat> &frames_gray, bool do_resize, const Size &size, bool to_gray) {
    Mat capture_frame;
    int cnt = 0;
    while (cnt < batch_maxsize) {
        if (use_frames) {
            if (cnt == frames_path.size())
                return false;
            capture_frame = imread(frames_path[cnt].string(), IMREAD_COLOR);
        } else {
            video_stream >> capture_frame;
            if (capture_frame.empty())
                return false;
        }

        Mat frame_gray;
        if (to_gray)
            cvtColor(capture_frame, frame_gray, COLOR_BGR2GRAY);
        else
            frame_gray = capture_frame.clone();
        if (do_resize) {
            Mat resized_frame_gray;
            resized_frame_gray.create(size, CV_8UC1);
            cv::resize(frame_gray, resized_frame_gray, size);
            frames_gray.push_back(resized_frame_gray);
        } else {
            frames_gray.push_back(frame_gray);
        }
        cnt++;
    }
    return true;
}

int DenseFlow::load_frames_video(VideoCapture &video_stream, vector<path> &frames_path, bool use_frames, bool do_resize,
                                 const Size &size, path output_dir, bool is_last, bool verbose) {
    int video_flow_idx = 0;
    vector<Mat> frames_gray_padding;
    while (true) {
        vector<Mat> frames_gray;
        bool is_open = load_frames_batch(video_stream, frames_path, use_frames, frames_gray, do_resize, size, true);
        vector<Mat> padded_frames_gray(frames_gray_padding.size() + frames_gray.size());
        copy(frames_gray_padding.begin(), frames_gray_padding.end(), padded_frames_gray.begin());
        copy(frames_gray.begin(), frames_gray.end(), padded_frames_gray.begin() + frames_gray_padding.size());
        FlowBuffer frames_gray_item(padded_frames_gray, output_dir, video_flow_idx, !is_open);
        unique_lock<mutex> lock(frames_gray_mtx);
        while (frames_gray_queue.size() == frames_gray_maxsize) {
            if (verbose)
                cout << "frames_gray_queue full, waiting..." << endl;
            cond_frames_gray_produce.wait(lock);
        }
        frames_gray_queue.push(frames_gray_item);
        if (verbose)
            cout << "push frames gray, video_flow_idx " << video_flow_idx << ", batch_size " << frames_gray.size()
                 << endl;
        cond_frames_gray_consume.notify_all();
        if (is_last && !is_open)
            ready_to_exit1 = true;
        lock.unlock();
        frames_gray_padding.resize(abs(step));
        copy(padded_frames_gray.end() - abs(step), padded_frames_gray.end(), frames_gray_padding.begin());

        int M = padded_frames_gray.size() - abs(step);
        video_flow_idx += M;
        // read done a video
        if (!is_open)
            break;
        if (use_frames) {
            frames_path.erase(frames_path.begin(), frames_path.begin() + M);
        }
    }
    return video_flow_idx + abs(step);
}

void DenseFlow::load_frames(bool use_frames, string save_type, bool verbose) {
    for (size_t i = 0; i < video_paths.size(); i++) {
        path video_path = video_paths[i];
        path output_dir = output_dirs[i];
        if (save_type == "h5") {
#if (USE_HDF5)
            //  create h5
            char h5_ext[1024];
            if (step > 1) {
                sprintf(h5_ext, "_p%d.h5", step);
            } else if (step < 0) {
                sprintf(h5_ext, "_m%d.h5", -step);
            } else {
                sprintf(h5_ext, ".h5");
            }
            string h5_file = output_dir.string() + h5_ext;
            // overwrite if exists
            hid_t file_id = H5Fcreate(h5_file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            herr_t status = H5Fclose(file_id);
            if (status < 0)
                throw std::runtime_error("Failed to save hdf5 file " + h5_file);
#else
            throw std::runtime_error("HDF5 support is not enabled, pls recompile");
#endif
        }
        VideoCapture video_stream;
        vector<path> frames_path;
        if (use_frames) {
            directory_iterator end_itr;
            for (directory_iterator itr(video_path); itr != end_itr; ++itr) {
                if (!is_regular_file(itr->status()) || itr->path().extension() != ".jpg")
                    continue;
                frames_path.push_back(itr->path());
            }
            if (frames_path.size() == 0) {
                if (verbose)
                    cout << video_path << " is empty!" << endl;
                continue;
            }
            sort(frames_path.begin(), frames_path.end());
        } else {
            video_stream.open(video_path.c_str());
            if (!video_stream.isOpened())
                throw std::runtime_error("cannot open video_path stream:" + video_path.string());
        }

        Size size;
        int frames_num;
        bool do_resize = get_new_size(video_stream, frames_path, use_frames, size, frames_num);
        if (verbose)
            cout << video_path << ", frames ≈ " << frames_num << endl;
        frames_num = load_frames_video(video_stream, frames_path, use_frames, do_resize, size, output_dir,
                                       i == video_paths.size() - 1, verbose);
        total_frames += frames_num; // exact frame count
        if (!use_frames)
            video_stream.release();
        if (verbose)
            cout << "loaded video " << video_path << ", " << frames_num << " frames" << endl;
    }
    if (verbose)
        cout << "load frames exit." << endl;
}

void DenseFlow::calc_optflows_imp(const FlowBuffer &frames_gray, const string &algorithm, int step, bool verbose,
                                  Stream &stream) {
    // create
    Ptr<cuda::FarnebackOpticalFlow> alg_farn;
    Ptr<cuda::OpticalFlowDual_TVL1> alg_tvl1;
    Ptr<cuda::BroxOpticalFlow> alg_brox;
#if (USE_NVFLOW)
    Ptr<NvidiaOpticalFlow_1_0> alg_nv;
#endif
    if (algorithm == "nv") {
#if (USE_NVFLOW)
        auto size = frames_gray.item_data[0].size();
        alg_nv = NvidiaOpticalFlow_1_0::create(size.width, size.height);
#else
        throw std::runtime_error("NV hardware flow not enabled, pls recompile");
#endif
    } else if (algorithm == "tvl1") {
        alg_tvl1 = cuda::OpticalFlowDual_TVL1::create();
    } else if (algorithm == "farn") {
        alg_farn = cuda::FarnebackOpticalFlow::create();
    } else if (algorithm == "brox") {
        alg_brox = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);
    }

    // compute
    int N = frames_gray.item_data.size();
    int M = max(N - abs(step), 0);
    vector<Mat> flows(M);
    if (M > 0) {
        GpuMat flow_gpu;
        GpuMat gray_a, gray_b;
        for (size_t i = 0; i < M; ++i) {
            Mat flow;
            int a = step > 0 ? i : i - step;
            int b = step > 0 ? i + step : i;
            gray_a.upload(frames_gray.item_data[a], stream);
            gray_b.upload(frames_gray.item_data[b], stream);
            if (algorithm == "nv") {
#if (USE_NVFLOW)
                auto size = frames_gray.item_data[0].size();
                alg_nv->calc(gray_a, gray_b, flow, stream);
                alg_nv->upSampler(flow, size.width, size.height, alg_nv->getGridSize(), flows[i]);
#endif
            } else {
                if (algorithm == "tvl1") {
                    alg_tvl1->calc(gray_a, gray_b, flow_gpu, stream);
                } else if (algorithm == "farn") {
                    alg_farn->calc(gray_a, gray_b, flow_gpu, stream);
                } else if (algorithm == "brox") {
                    GpuMat d_buf_0, d_buf_1;
                    gray_a.convertTo(d_buf_0, CV_32F, 1.0 / 255.0, stream);
                    gray_b.convertTo(d_buf_1, CV_32F, 1.0 / 255.0, stream);
                    alg_brox->calc(d_buf_0, d_buf_1, flow_gpu, stream);
                } else {
                    throw std::runtime_error("unknown optical algorithm " + algorithm);
                    return;
                }
                flow_gpu.download(flows[i]);
                total_flows += 1;
            }
        }

        // release
        if (algorithm == "nv") {
#if (USE_NVFLOW)
            alg_nv.release();
#endif
        } else if (algorithm == "tvl1") {
            alg_tvl1.release();
        } else if (algorithm == "farn") {
            alg_farn.release();
        } else if (algorithm == "brox") {
            alg_brox.release();
        }
    }

    FlowBuffer flow_buffer(flows, frames_gray.output_dir, frames_gray.base_start, frames_gray.last_buffer);
    unique_lock<mutex> lock(flows_mtx);
    while (flows_queue.size() == flows_maxsize) {
        if (verbose)
            cout << "flows queue is full, waiting..." << endl;
        cond_flows_produce.wait(lock);
    }
    flows_queue.push(flow_buffer);
    if (verbose)
        cout << "flows queue push a item, size " << flows_queue.size() << endl;
    cond_flows_consume.notify_all();
    lock.unlock();
}

void DenseFlow::calc_optflows(bool verbose) {
    while (true) {
        unique_lock<mutex> lock(frames_gray_mtx);
        while (frames_gray_queue.size() == 0) {
            if (verbose)
                cout << "frames_gray_queue empty, waiting..." << endl;
            cond_frames_gray_consume.wait(lock);
        }
        FlowBuffer frames_gray = frames_gray_queue.front();
        frames_gray_queue.pop();
        if (ready_to_exit1 && frames_gray_queue.size() == 0)
            ready_to_exit2 = true;
        cond_frames_gray_produce.notify_all();
        lock.unlock();
        calc_optflows_imp(frames_gray, algorithm, step, false, stream);
        if (ready_to_exit2)
            break;
        else
            std::this_thread::yield();
    }
    if (verbose)
        cout << "calc optflows exit." << endl;
}

void DenseFlow::encode_save(string save_type, bool verbose) {
    while (true) {
        unique_lock<mutex> lock(flows_mtx);
        while (flows_queue.size() == 0) {
            if (verbose)
                cout << "flows queue empty, waiting..." << endl;
            cond_flows_consume.wait(lock);
        }
        FlowBuffer flow_buffer = flows_queue.front();
        flows_queue.pop();
        if (ready_to_exit2 && flows_queue.size() == 0)
            ready_to_exit3 = true;
        if (verbose)
            cout << "flows queue get a item, size " << flows_queue.size() << endl;
        cond_flows_produce.notify_all();
        lock.unlock();
        // encode & save
        int M = flow_buffer.item_data.size();
        if (save_type == "jpg") {
            vector<vector<uchar>> output_x, output_y;
            Mat planes[2];
            for (int i = 0; i < M; ++i) {
                split(flow_buffer.item_data[i], planes);
                Mat flow_x(planes[0]);
                Mat flow_y(planes[1]);
                vector<uchar> str_x, str_y;
                encodeFlowMap(flow_x, flow_y, str_x, str_y, bound);
                output_x.push_back(str_x);
                output_y.push_back(str_y);
            }
            writeFlowImages(output_x, (flow_buffer.output_dir / "flow_x").c_str(), step, flow_buffer.base_start);
            writeFlowImages(output_y, (flow_buffer.output_dir / "flow_y").c_str(), step, flow_buffer.base_start);
        } else if (save_type == "h5") {
#if (USE_HDF5)
            vector<Mat> output_h5_x, output_h5_y;
            for (int i = 0; i < M; ++i) {
                Mat planes[2];
                split(flow_buffer.item_data[i], planes);
                Mat flow_x(planes[0]);
                Mat flow_y(planes[1]);
                output_h5_x.push_back(flow_x);
                output_h5_y.push_back(flow_y);
            }
            writeHDF5(output_h5_x, flow_buffer.output_dir.c_str(), "flow_x", step, flow_buffer.base_start);
            writeHDF5(output_h5_y, flow_buffer.output_dir.c_str(), "flow_y", step, flow_buffer.base_start);
#endif
        } else if (save_type == "png") {
            vector<vector<uchar>> output;
            for (int i = 0; i < M; ++i) {
                Mat planes[2];
                split(flow_buffer.item_data[i], planes);
                Mat flow_x(planes[0]);
                Mat flow_y(planes[1]);
                vector<uchar> str;
                encodeFlowMapPng(flow_x, flow_y, str);
                output.push_back(str);
            }
            writeFlowImagesPng(output, (flow_buffer.output_dir / "flow").c_str(), step, flow_buffer.base_start);
        }
        // mark done for the last batch of the video
        if (is_record && flow_buffer.last_buffer) {
            path donedir;
            path title;
            if (has_class) {
                donedir = flow_buffer.output_dir.parent_path().parent_path() / ".done" /
                          flow_buffer.output_dir.parent_path().filename();
                title = flow_buffer.output_dir.parent_path().filename() / flow_buffer.output_dir.filename();
            } else {
                donedir = flow_buffer.output_dir.parent_path() / ".done";
                title = flow_buffer.output_dir.filename();
            }
            path donefile = donedir / flow_buffer.output_dir.stem().string();
            createFile(donefile);
            cout << "done video " << title << endl;
        }

        if (ready_to_exit3)
            break;
    }
    if (verbose)
        cout << "post process exit." << endl;
}

void calcDenseFlowVideoGPU(vector<path> video_paths, vector<path> output_dirs, string algorithm, int step, int bound,
                           int new_width, int new_height, int new_short, bool has_class, bool use_frames,
                           string save_type, bool is_record, bool verbose) {
    setDevice(0);
    DenseFlow flow_video_gpu(video_paths, output_dirs, algorithm, step, bound, new_width, new_height, new_short,
                             has_class, is_record, save_type);
    double start_t = CurrentSeconds();
    if (step == 0) {
        flow_video_gpu.extract_frames_only(use_frames, verbose);
    } else {
        flow_video_gpu.launch(use_frames, save_type, verbose);
    }
    double end_t = CurrentSeconds();
    unsigned long N = flow_video_gpu.get_processed_total_frames();
    unsigned long F = flow_video_gpu.get_processed_total_flows();
    cout << video_paths.size() << " videos (" << N << " frames, " << F << " " << algorithm
         << " flows) processed, using " << end_t - start_t << "s, decoding speed " << N / (end_t - start_t)
         << "fps, flow speed " << F / (end_t - start_t) << "fps" << endl;
}
