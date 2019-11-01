#include "dense_flow.h"
#include "utils.h"
#include <clue/stringex.hpp>
#include <clue/textio.hpp>
#include <clue/thread_pool.hpp>
#include <boost/filesystem.hpp>

INITIALIZE_EASYLOGGINGPP

using namespace cv::cuda;
using clue::ends_with;
using clue::line_stream;
using clue::read_file_content;
using clue::string_view;
using clue::thread_pool;
using boost::filesystem::path;
using boost::filesystem::create_directories;

int main(int argc, char **argv) {
    try {
        const char *keys = {"{ v video     | video.mp4        | filename of video }"
                            "{ o outputDir | /path/to/outputs | root dir of output }"
                            "{ a algorithm | nv               | optical flow algorithm (nv/tvl1/farn/brox) }"
                            "{ s step      | 1                | right - left (0 for img, non-0 for flow) }"
                            "{ b bound     | 32               | maximum of optical flow }"
                            "{ w newWidth  | 0                | new width }"
                            "{ h newHeight | 0                | new height }"
                            "{ sh short    | 0                | short side length }"
                            "{ d deviceId  | 0                | set gpu id }"
                            "{ p parallel  | 1                | parallel threads }"
                            "{ vv verbose  |                  | verbose }"
                            "{ help        |                  | print help message }"};

        CommandLineParser cmd(argc, argv, keys);
        string video_path = cmd.get<string>("video");
        string output_dir = cmd.get<string>("outputDir");
        string algorithm = cmd.get<string>("algorithm");
        int step = cmd.get<int>("step");
        int bound = cmd.get<int>("bound");
        int new_width = cmd.get<int>("newWidth");
        int new_height = cmd.get<int>("newHeight");
        int new_short = cmd.get<int>("short");
        int device_id = cmd.get<int>("deviceId");
        int parallel = cmd.get<int>("parallel");
        bool verbose = cmd.has("verbose");

        cmd.about("GPU optical flow extraction.");
        if (cmd.has("help") || !cmd.check()) {
            cmd.printMessage();
            cmd.printErrors();
            return 0;
        }

        if (ends_with(video_path, ".txt")) {
            string text = read_file_content(video_path);
            thread_pool P(parallel);
            line_stream lstr(text);
            for (string_view line : lstr) {
                P.schedule([&](size_t tid) {
                    string v = line.to_string();
                    path p(v);
                    path out = path(output_dir) / p.stem();
                    create_directories(out);
                    calcDenseNvFlowVideoGPU(v, out.c_str(), algorithm, step, bound, new_width, new_height,
                                            new_short, device_id, verbose);
                });
            }
            P.wait_done();
        } else {
            calcDenseNvFlowVideoGPU(video_path, output_dir, algorithm, step, bound, new_width, new_height, new_short,
                                    device_id, verbose);
        }

    } catch (const std::exception &ex) {
        std::cout << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
