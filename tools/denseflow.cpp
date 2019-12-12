#include "dense_flow.h"
#include "opencv2/opencv.hpp"
#include "utils.h"
#include <fstream>

int main(int argc, char **argv) {
    try {
        const char *keys = {"{ h help         |      | print help message }"
                            "{ @input         |      | filename of video or folder of frames or a list.txt of those }"
                            "{ o outputDir    | .    | root dir of output }"
                            "{ a algorithm    | tvl1 | optical flow algorithm (nv/tvl1/farn/brox) }"
                            "{ s step         | 1    | right - left (0 for img, non-0 for flow) }"
                            "{ b bound        | 32   | maximum of optical flow }"
                            "{ nw newWidth    | 0    | new width }"
                            "{ nh newHeight   | 0    | new height }"
                            "{ ns newShort    | 0    | short side length }"
                            "{ d deviceId     | 0    | set gpu id }"
                            "{ cf classFolder |      | outputDir/class/video/flow.jpg }"
                            "{ if inputFrames |      | inputs are frames }"
                            "{ v verbose      |      | verbose }"};

        CommandLineParser cmd(argc, argv, keys);

        cmd.about("GPU optical flow extraction.");
        if (cmd.get<string>("@input") == "" || cmd.has("help")) {
            cmd.printMessage();
            return 0;
        }
        if (!cmd.check()) {
            cmd.printErrors();
            return 0;
        }

        path video_path(cmd.get<string>("@input"));
        path output_dir(cmd.get<string>("outputDir"));
        string algorithm = cmd.get<string>("algorithm");
        int step = cmd.get<int>("step");
        int bound = cmd.get<int>("bound");
        int new_width = cmd.get<int>("newWidth");
        int new_height = cmd.get<int>("newHeight");
        int new_short = cmd.get<int>("newShort");
        int device_id = cmd.get<int>("deviceId");
        bool has_class = cmd.has("classFolder");
        bool use_frames = cmd.has("inputFrames");
        bool verbose = cmd.has("verbose");

        Mat::setDefaultAllocator(HostMem::getAllocator(HostMem::AllocType::PAGE_LOCKED));

        vector<path> video_paths;
        vector<path> output_dirs;
        bool is_record = false;
        if (video_path.extension() == ".txt") {
            is_record = true;
            std::ifstream ifs(video_path.string());
            string line;
            while (getline(ifs, line)) {
                path vidfile(line);
                path outdir;
                if (has_class) {
                    outdir = output_dir / vidfile.parent_path().filename() / vidfile.stem();
                } else {
                    outdir = output_dir / vidfile.stem();
                }
                create_directories(outdir);
                video_paths.push_back(vidfile);
                output_dirs.push_back(outdir);
                // mark done
                path donedir;
                if (has_class) {
                    donedir = output_dir / ".done" / vidfile.parent_path().filename();
                } else {
                    donedir = output_dir / ".done";
                }
                create_directories(donedir);
            }
        } else {
            path outdir = output_dir / video_path.stem();
            create_directories(outdir);
            video_paths.push_back(video_path);
            output_dirs.push_back(outdir);
        }
        calcDenseFlowVideoGPU(video_paths, output_dirs, algorithm, step, bound, new_width, new_height, new_short,
                              has_class, device_id, use_frames, is_record, verbose);

    } catch (const std::exception &ex) {
        std::cout << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
