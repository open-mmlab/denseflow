//
// Created by alex on 16-3-27.
//

#include "dense_flow.h"
#include "utils.h"
#include <clue/timing.hpp>
#include <clue/textio.hpp>
#include <random>
#include "easylogging++.h"

INITIALIZE_EASYLOGGINGPP

using namespace cv::gpu;

void OpenVideoOnly(string video_path){
    VideoCapture video_stream(video_path);
    CHECK(video_stream.isOpened())<<"Cannot open video stream \""
                                  <<video_path<<"\"";
    video_stream.release();
}

std::pair<double, double> SeekByFrame(string video_path, vector<float> ratio, int read_num, double& first_seek_time){
    VideoCapture video_stream(video_path);

    double seeking_time = 0, reading_time = 0;
    int cnt = 0;
    for (float r : ratio) {
        CHECK(video_stream.isOpened())<<"Cannot open video stream \""
                                      <<video_path<<"\"";
        int frame_cnt = (int) video_stream.get(CV_CAP_PROP_FRAME_COUNT);
        int frame_idx = (int) (frame_cnt * r);

        auto sw = clue::stop_watch(true);
        video_stream.set(CV_CAP_PROP_POS_FRAMES, frame_idx);

        double seek = sw.elapsed().msecs();
        seeking_time += seek;

        sw.reset();
        sw.start();
        Mat capture_frame;

        for (int i = 0; i < read_num; ++i) {
            video_stream >> capture_frame;
        }
        reading_time += sw.elapsed().msecs();

        if (cnt == 0){
            first_seek_time = seeking_time;
            LOG(INFO)<<"first seek "<<seek;
        }else {
            LOG(INFO)<<"consequtive seek "<<seek;
        }
        cnt ++;
    }
    video_stream.release();

    return std::make_pair(seeking_time, reading_time);
}

void SeekByMSEC(string video_path, float ratio, int read_num){
    VideoCapture video_stream(video_path);
    CHECK(video_stream.isOpened())<<"Cannot open video stream \""
                                  <<video_path<<"\"";
    int frame_cnt = (int)video_stream.get(CV_CAP_PROP_FRAME_COUNT);
    int fps = (int)video_stream.get(CV_CAP_PROP_FPS);
    float time_msec = 1000 * frame_cnt / fps * ratio;

    video_stream.set(CV_CAP_PROP_POS_MSEC, time_msec);
    Mat capture_frame;

    for(int i = 0; i < read_num; ++i){
        video_stream>>capture_frame;
    }
    video_stream.release();
}

void SeekByRATIO(string video_path, float ratio, int read_num){
    VideoCapture video_stream(video_path);
    CHECK(video_stream.isOpened())<<"Cannot open video stream \""
                                  <<video_path<<"\"";

    video_stream.set(CV_CAP_PROP_POS_AVI_RATIO, ratio);
    Mat capture_frame;

    for(int i = 0; i < read_num; ++i){
        video_stream>>capture_frame;
    }
    video_stream.release();
}

int main(int argc, char** argv){

    CHECK(argc==6)<<"Need a input video list to test";

    //read video list
    vector<string> video_list;
    std::ifstream list_file(argv[1]);
    CHECK(list_file.is_open());

    int read_num;
    clue::try_parse(argv[2], read_num);

    string vid_path = string(argv[3]);

    while(list_file.good()){
        string name;
        list_file>>name;
        video_list.push_back(name);
    }
    list_file.close();

    int test_limit;
    clue::try_parse(argv[4], test_limit);

    int seek_num;
    clue::try_parse(argv[5], seek_num);

    LOG(INFO)<<"Video list loaded, "<<video_list.size()<<" video in total.";

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.01, 0.99);

    std::shuffle(video_list.begin(), video_list.end(), gen);

    // time seeking by frame number
    LOG(INFO)<<"Timing started. Reading "<<read_num<<" frame(s) per video. Seeking "<<seek_num<<" times ";
    auto aw = clue::stop_watch(true);

    int cnt = 0;
    double accum_seeking_time=0, accum_reading_time=0, accum_first_seeking_time=0;
    for(auto video_id : video_list){
        string video_full_path = vid_path + video_id;
        vector<float> ratios;
        for (int i = 0; i < seek_num; i++){
            ratios.push_back(dist(gen));
        }
        double fs;
        auto pair = SeekByFrame(video_full_path, ratios, read_num, fs);
        accum_seeking_time += pair.first;
        accum_reading_time += pair.second;
        accum_first_seeking_time += fs;
        cnt ++;
        if (cnt >= test_limit){
            break;
        }
    }

    double elapsed = aw.elapsed().msecs();
    double avg_time = elapsed / cnt;

    LOG(INFO)<<"Seeking by Frame Finished. Total time "<<elapsed<<" ms. Average "<<avg_time<<" msec/video."
            <<" seeking time "<<accum_seeking_time/cnt<<" msec/video"<<". reading time "<<accum_reading_time/cnt<<" msec/video"
            <<". \n first seeking time "<<accum_first_seeking_time/cnt;
    return 0;
}