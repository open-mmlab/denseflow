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

void drawOptFlowMap(const Mat &flow, Mat &cflowmap, int step, double, const Scalar &color) {
    for (int y = 0; y < cflowmap.rows; y += step)
        for (int x = 0; x < cflowmap.cols; x += step) {
            const Point2f &fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x, y), Point(cvRound(x + fxy.x), cvRound(y + fxy.y)), color);
            circle(cflowmap, Point(x, y), 2, color, -1);
        }
}

void encodeFlowMap(const Mat &flow_map_x, const Mat &flow_map_y, std::vector<uchar> &encoded_x,
                   std::vector<uchar> &encoded_y, int bound, bool to_jpg) {
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

void writeImages(std::vector<std::vector<uchar>> images, std::string name_temp, const int base) {
    for (int i = base; i < images.size() + base; ++i) {
        char tmp[256];
        sprintf(tmp, "_%05d.jpg", i + 1);
        FILE *fp;
        fp = fopen((name_temp + tmp).c_str(), "wb");
        fwrite(images[i].data(), 1, images[i].size(), fp);
        fclose(fp);
    }
}

void writeImagesV2(std::vector<std::vector<uchar>> images, std::vector<std::vector<uchar>> flow_x,
                   std::vector<std::vector<uchar>> flow_y, std::vector<std::string> names,
                   const string &output_root_dir) {
    for (int i = 0; i < names.size(); i++) {
        // xx.jpg as default image formate
        // std::cout << output_root_dir+"/flow_i_"+names[i]<< std::endl;
        FILE *fp_img, *fp_x, *fp_y;
        fp_img = fopen((output_root_dir + "/flow_i_" + names[i]).c_str(), "wb");
        fp_x = fopen((output_root_dir + "/flow_x_" + names[i]).c_str(), "wb");
        fp_y = fopen((output_root_dir + "/flow_y_" + names[i]).c_str(), "wb");
        fwrite(images[i].data(), 1, images[i].size(), fp_img);
        fwrite(flow_x[i].data(), 1, flow_x[i].size(), fp_x);
        fwrite(flow_y[i].data(), 1, flow_y[i].size(), fp_y);
        fclose(fp_img);
        fclose(fp_x);
        fclose(fp_y);
    }
}
