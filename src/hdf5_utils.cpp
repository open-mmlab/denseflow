#include "hdf5_utils.hpp"

void hdf5_save_nd_dataset(const hid_t file_id, const string &dataset_name, const cv::Mat &mat) {
    // std::cout << "type: " << mat.type() << ", rows: " << mat.rows << ", cols: " << mat.cols
    //           << ", channels:" << mat.channels() << std::endl;

    int num_axes = mat.channels() == 1 ? 2 : mat.channels();
    hsize_t *dims = new hsize_t[num_axes];
    dims[0] = mat.rows;
    dims[1] = mat.cols;
    if (num_axes == 3)
        dims[2] = 3;

    // float* data = mat.ptr<float>();
    herr_t status = H5LTmake_dataset_float(file_id, dataset_name.c_str(), num_axes, dims, mat.ptr<float>());
    if (status < 0) {
        throw "Failed to make float dataset " + dataset_name;
    }
    delete[] dims;
}
