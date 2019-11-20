#include "utils.h"

void SplitString(const std::string &s, std::vector<std::string> &v, const std::string &c) {
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while (std::string::npos != pos2) {
        v.push_back(s.substr(pos1, pos2 - pos1));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if (pos1 != s.length())
        v.push_back(s.substr(pos1));
}

void createFile(const path &ph) {
    std::ofstream f(ph.string());
    f.close();
}

double CurrentSeconds() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
               .count() /
           1000.0;
}
#if (USE_HDF5)
void hdf5_save_nd_dataset(const hid_t file_id, const string &dataset_name, const cv::Mat &mat) {
    int num_axes = mat.channels() == 1 ? 2 : mat.channels();
    hsize_t *dims = new hsize_t[num_axes];
    dims[0] = mat.rows;
    dims[1] = mat.cols;
    if (num_axes == 3)
        dims[2] = 3;
    herr_t status = H5LTmake_dataset_float(file_id, dataset_name.c_str(), num_axes, dims, mat.ptr<float>());
    if (status < 0) {
        throw "Failed to make float dataset " + dataset_name;
    }
    delete[] dims;
}
#endif
