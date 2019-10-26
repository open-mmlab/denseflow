#include "dense_flow.h"
#include "utils.h"

INITIALIZE_EASYLOGGINGPP

using namespace cv::cuda;

int main(int argc, char **argv) {
    // IO operation
    const char *keys = {"{ f fileList       | imglist.txt              | filename of imglist }"
                        "{ r rootDir        | /path/to/images/root_dir | root dir of imglist }"
                        "{ or outputRootDir | /path/to/output/root_dir | root dir of output }"
                        "{ b bound          | 15                       | specify the maximum of optical flow}"
                        "{ t type           | 0                        | specify the optical flow algorithm }"
                        "{ d device_id      | 0                        | set gpu id}"
                        "{ o out            | zip                      | output style}"
                        "{ w newWidth       | 0                        | output style}"
                        "{ h newHeight      | 0                        | output style}"};

    CommandLineParser cmd(argc, argv, keys);
    std::string fileList = cmd.get<std::string>("fileList");
    std::string rootDir = cmd.get<std::string>("rootDir");
    std::string outputRootDir = cmd.get<std::string>("outputRootDir");
    std::string output_style = cmd.get<std::string>("out");
    int bound = cmd.get<int>("bound");
    int type = cmd.get<int>("type");
    int device_id = cmd.get<int>("device_id");
    int new_height = cmd.get<int>("newHeight");
    int new_width = cmd.get<int>("newWidth");

    calcDenseFlowFramesGPU(fileList, rootDir, outputRootDir, bound, type, device_id, new_width, new_height);

    /*
            if (output_style == "dir") {
                    writeImages(out_vec_x, xFlowFile);
                    writeImages(out_vec_y, yFlowFile);
                    writeImages(out_vec_img, imgFile);
            }else{
    //		LOG(INFO)<<"Writing results to Zip archives";
                    writeZipFile(out_vec_x, "x_%05d.jpg", xFlowFile+".zip");
                    writeZipFile(out_vec_y, "y_%05d.jpg", yFlowFile+".zip");
                    writeZipFile(out_vec_img, "img_%05d.jpg", imgFile+".zip");
            }
    */
    return 0;
}
