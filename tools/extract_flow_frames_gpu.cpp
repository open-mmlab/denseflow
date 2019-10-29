#include "dense_flow.h"
#include "utils.h"

INITIALIZE_EASYLOGGINGPP

using namespace cv::cuda;

int main(int argc, char **argv) {
    // IO operation
    const char *keys = {"{ f fileList       | imglist.txt              | filename of imglist }"
                        "{ r rootDir        | /path/to/images/root_dir | root dir of imglist }"
                        "{ or outputRootDir | /path/to/output/root_dir | root dir of output }"
                        "{ b bound          | 20                       | maximum of optical flow }"
                        "{ t type           | 1                        | optical flow algorithm }"
                        "{ d deviceId       | 0                        | set gpu id }"
                        "{ si saveImg       | 1                        | save images }"
                        "{ sj saveJpg       | 1                        | save flow images }"
                        "{ sh saveH5        | 0                        | save h5 }"
                        "{ sz saveZip       | 0                        | save zipped flow }"
                        "{ w newWidth       | 0                        | output style }"
                        "{ h newHeight      | 0                        | output style }"};

    CommandLineParser cmd(argc, argv, keys);
    std::string fileList = cmd.get<std::string>("fileList");
    std::string rootDir = cmd.get<std::string>("rootDir");
    std::string outputRootDir = cmd.get<std::string>("outputRootDir");
    bool save_img = cmd.get<bool>("saveImg");
    bool save_jpg = cmd.get<bool>("saveJpg");
    bool save_h5 = cmd.get<bool>("saveH5");
    bool save_zip = cmd.get<bool>("saveZip");
    int bound = cmd.get<int>("bound");
    int type = cmd.get<int>("type");
    int device_id = cmd.get<int>("deviceId");
    int new_height = cmd.get<int>("newHeight");
    int new_width = cmd.get<int>("newWidth");

    calcDenseFlowFramesGPU(fileList, rootDir, outputRootDir, bound, type, device_id, new_width, new_height, save_img, save_jpg, save_h5, save_zip);

    return 0;
}
