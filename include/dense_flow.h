#ifndef DENSEFLOW_DENSE_FLOW_H
#define DENSEFLOW_DENSE_FLOW_H

#include "common.h"

void calcDenseFlowVideoGPU(vector<path> video_paths, vector<path> output_dirs, string algorithm, int step, int bound,
                           int new_width, int new_height, int new_short, bool has_class, bool use_frames,
                           string save_type, bool is_record, bool verbose);

class FlowBuffer {
  public:
    vector<Mat> item_data;
    path output_dir;
    int base_start;
    bool last_buffer;
    FlowBuffer(vector<Mat> item_data, path output_dir, int base_start, bool last_buffer)
        : item_data(item_data), output_dir(output_dir), base_start(base_start), last_buffer(last_buffer) {}
};

class DenseFlow {
  private:
    vector<path> video_paths;
    vector<path> output_dirs;
    string algorithm;
    string save_type;
    int step;
    int bound;
    int new_width;
    int new_height;
    int new_short;
    bool has_class;
    bool is_record;
    Stream stream;

    mutex frames_gray_mtx, flows_mtx;
    condition_variable cond_frames_gray_produce, cond_frames_gray_consume;
    condition_variable cond_flows_produce, cond_flows_consume;
    int frames_gray_maxsize;
    int flows_maxsize;
    int batch_maxsize;
    bool ready_to_exit1;
    bool ready_to_exit2;
    bool ready_to_exit3;

    queue<FlowBuffer> frames_gray_queue;
    queue<FlowBuffer> flows_queue;
    unsigned long total_frames;
    unsigned long total_flows;

    // meber functions
    bool check_param();
    bool get_new_size(const VideoCapture &video_stream, const vector<path> &frames_path, bool use_frames,
                      Size &new_size, int &frames_num);
    bool load_frames_batch(VideoCapture &video_stream, const vector<path> &frames_path, bool use_frames,
                           vector<Mat> &frames_gray, bool do_resize, const Size &size, bool to_gray);
    int load_frames_video(VideoCapture &video_stream, vector<path> &frames_path, bool use_frames, bool do_resize,
                          const Size &size, path output_dir, bool is_last, bool verbose);
    void calc_optflows_imp(const FlowBuffer &frames_gray, const string &algorithm, int step, bool verbose,
                           Stream &stream = Stream::Null());
    void load_frames(bool use_frames, string save_type, bool verbose = true);
    void calc_optflows(bool verbose = true);
    void encode_save(string save_type, bool verbose = true);
    int extract_frames_video(VideoCapture &video_stream, vector<path> &frames_path, bool use_frames, bool do_resize,
                             const Size &size, path output_dir, bool verbose);

  public:
    static void load_frames_wrap(void *arg, bool use_frames, string save_type, bool verbose) {
        return static_cast<DenseFlow *>(arg)->load_frames(use_frames, save_type, verbose);
    }
    static void calc_optflows_wrap(void *arg, bool verbose) {
        return static_cast<DenseFlow *>(arg)->calc_optflows(verbose);
    }
    static void encode_save_wrap(void *arg, string save_type, bool verbose) {
        return static_cast<DenseFlow *>(arg)->encode_save(save_type, verbose);
    }
    void launch(bool use_frames, string save_type, bool verbose) {
        thread thread_load_frames(load_frames_wrap, this, use_frames, save_type, verbose);
        thread thread_calc_optflow(calc_optflows_wrap, this, false);
        thread thread_encode_save(encode_save_wrap, this, save_type, false);
        thread_load_frames.join();
        thread_calc_optflow.join();
        thread_encode_save.join();
    }
    void extract_frames_only(bool use_frames, bool verbose);
    unsigned long get_processed_total_frames() { return total_frames; }
    unsigned long get_processed_total_flows() { return total_flows; }

    DenseFlow(vector<path> video_paths, vector<path> output_dirs, string algorithm, int step, int bound, int new_width,
              int new_height, int new_short, bool has_class, bool is_record, string save_type)
        : video_paths(video_paths), output_dirs(output_dirs), algorithm(algorithm), step(step), bound(bound),
          new_width(new_width), new_height(new_height), new_short(new_short), has_class(has_class),
          is_record(is_record), save_type(save_type) {
        if (!check_param())
            throw std::runtime_error("check init param error.");
        batch_maxsize = 512;
        frames_gray_maxsize = flows_maxsize = 3;
        ready_to_exit1 = ready_to_exit2 = ready_to_exit3 = false;
        total_frames = 0;
        total_flows = 0;
    }
};

#endif // DENSEFLOW_DENSE_FLOW_H
