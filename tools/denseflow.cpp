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
                            "{ s step         | 0    | right - left (0 for img, non-0 for flow) }"
                            "{ b bound        | 32   | maximum of optical flow }"
                            "{ nw newWidth    | 0    | new width }"
                            "{ nh newHeight   | 0    | new height }"
                            "{ ns newShort    | 0    | short side length }"
                            "{ cf classFolder |      | outputDir/class/video/flow.jpg }"
                            "{ if inputFrames |      | inputs are frames }"
                            "{ st saveType    | jpg  | save format type (png/h5/jpg) }"
                            "{ f force        |      | regardless of the marked .done file }"
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
        bool has_class = cmd.has("classFolder");
        bool use_frames = cmd.has("inputFrames");
        bool force = cmd.has("force");
        string save_type = cmd.get<string>("saveType");
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
                path donedir;
                path donefile;
                if (has_class) {
                    outdir = output_dir / vidfile.parent_path().filename() / vidfile.stem();
                    donedir = output_dir / ".done" / vidfile.parent_path().filename();
                } else {
                    outdir = output_dir / vidfile.stem();
                    donedir = output_dir / ".done";
                }
                donefile = donedir / vidfile.stem();
                if (!force && is_regular_file(donefile)) {
                    if (verbose) {
                        cout << "skip " << vidfile.parent_path().filename() / vidfile.stem() << endl;
                    }
                    continue;
                }
                create_directories(outdir);
                create_directories(donedir);
                video_paths.push_back(vidfile);
                output_dirs.push_back(outdir);
            }
        } else {
            path outdir = output_dir / video_path.stem();
            create_directories(outdir);
            video_paths.push_back(video_path);
            output_dirs.push_back(outdir);
        }
        if (video_paths.size() > 0) {
            calcDenseFlowVideoGPU(video_paths, output_dirs, algorithm, step, bound, new_width, new_height, new_short,
                                  has_class, use_frames, save_type, is_record, verbose);
        }

    } catch (const std::exception &ex) {
        cout << ex.what() << endl;
        return 1;
    }
    return 0;
}
