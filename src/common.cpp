//
// Created by yjxiong on 11/18/15.
//

#include "common.h"

void convertFlowToImage(const Mat &flow_x, const Mat &flow_y, Mat &img_x, Mat &img_y, double lowerBound,
                        double higherBound) {
#define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255 * ((v) - (L)) / ((H) - (L))))
    for (int i = 0; i < flow_x.rows; ++i) {
        for (int j = 0; j < flow_y.cols; ++j) {
            float x = flow_x.at<float>(i, j);
            float y = flow_y.at<float>(i, j);
            img_x.at<uchar>(i, j) = CAST(x, lowerBound, higherBound);
            img_y.at<uchar>(i, j) = CAST(y, lowerBound, higherBound);
        }
    }
#undef CAST
}

void encodeFlowMap(const Mat &flow_map_x, const Mat &flow_map_y, vector<uchar> &encoded_x, vector<uchar> &encoded_y,
                   int bound, bool to_jpg) {
    Mat flow_img_x(flow_map_x.size(), CV_8UC1);
    Mat flow_img_y(flow_map_y.size(), CV_8UC1);

    convertFlowToImage(flow_map_x, flow_map_y, flow_img_x, flow_img_y, -bound, bound);

    if (to_jpg) {
        imencode(".jpg", flow_img_x, encoded_x);
        imencode(".jpg", flow_img_y, encoded_y);
    } else {
        encoded_x.resize(flow_img_x.total());
        encoded_y.resize(flow_img_y.total());
        memcpy(encoded_x.data(), flow_img_x.data, flow_img_x.total());
        memcpy(encoded_y.data(), flow_img_y.data, flow_img_y.total());
    }
}

void writeImages(vector<vector<uchar>> images, string name_prefix) {
    for (int i = 0; i < images.size(); ++i) {
        char tmp[256];
        sprintf(tmp, "_%05d.jpg", i);
        FILE *fp;
        fp = fopen((name_prefix + tmp).c_str(), "wb");
        fwrite(images[i].data(), 1, images[i].size(), fp);
        fclose(fp);
    }
}

void writeFlowImages(vector<vector<uchar>> images, string name_prefix, const int step) {
    int base = step > 0 ? 0 : -step;
    for (int i = 0; i < images.size(); ++i) {
        char tmp[256];
        if (step > 1) {
            sprintf(tmp, "_p%d_%05d.jpg", step, i + base);
        } else if (step < 0) {
            sprintf(tmp, "_m%d_%05d.jpg", -step, i + base);
        } else {
            sprintf(tmp, "_%05d.jpg", i + base);
        }
        FILE *fp;
        fp = fopen((name_prefix + tmp).c_str(), "wb");
        fwrite(images[i].data(), 1, images[i].size(), fp);
        fclose(fp);
    }
}
