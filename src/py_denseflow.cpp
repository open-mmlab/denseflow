#include <iostream>
#include <boost/python.hpp>
#include <Python.h>
#include <vector>


#include "common.h"
#include "opencv2/gpu/gpu.hpp"
using namespace cv::gpu;
using namespace cv;

namespace bp = boost::python;

class TVL1FlowExtractor{
public:

    TVL1FlowExtractor(int bound){
        bound_ = bound;
    }

    static void set_device(int dev_id){
        setDevice(dev_id);
    }

    bp::list extract_flow(bp::list frames, int img_width, int img_height){
        bp::list output;
        Mat input_frame, prev_frame, next_frame, prev_gray, next_gray;
        Mat flow_x, flow_y;




        // initialize the first frame
        const char* first_data = ((const char*)bp::extract<const char*>(frames[0]));
        input_frame = Mat(img_height, img_width, CV_8UC3);
        initializeMats(input_frame, prev_frame, prev_gray, next_frame, next_gray);

        memcpy(prev_frame.data, first_data, bp::len(frames[0]));
        cvtColor(prev_frame, prev_gray, CV_BGR2GRAY);
        for (int idx = 1; idx < bp::len(frames); idx++){
            const char* this_data = ((const char*)bp::extract<const char*>(frames[idx]));
            memcpy(next_frame.data, this_data, bp::len(frames[0]));
            cvtColor(next_frame, next_gray, CV_BGR2GRAY);

            d_frame_0.upload(prev_gray);
            d_frame_1.upload(next_gray);

            alg_tvl1(d_frame_0, d_frame_1, d_flow_x, d_flow_y);

            d_flow_x.download(flow_x);
            d_flow_y.download(flow_y);

            vector<uchar> str_x, str_y;

            encodeFlowMap(flow_x, flow_y, str_x, str_y, bound_, false);
            output.append(
                bp::make_tuple(
                    bp::str((const char*) str_x.data(), str_x.size()),
                    bp::str((const char*) str_y.data(), str_y.size())
                    )
            );

            std::swap(prev_gray, next_gray);
        }
        return output;
    };
private:
    int bound_;
    GpuMat d_frame_0, d_frame_1;
    GpuMat d_flow_x, d_flow_y;
    OpticalFlowDual_TVL1_GPU alg_tvl1;
};


//// Boost Python Related Decl
BOOST_PYTHON_MODULE(libpydenseflow){
    bp::class_<TVL1FlowExtractor>("TVL1FlowExtractor", bp::init<int>())
            .def("extract_flow", &TVL1FlowExtractor::extract_flow)
            .def("set_device", &TVL1FlowExtractor::set_device)
            .staticmethod("set_device");
}