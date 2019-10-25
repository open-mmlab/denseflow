#ifndef _HDF5_UTILS_HPP
#define _HDF5_UTILS_HPP
#include "hdf5.h"
#include "hdf5_hl.h"
#include "common.h"

void hdf5_save_nd_dataset(const hid_t file_id, const string& dataset_name, const Mat& mat);

#endif 