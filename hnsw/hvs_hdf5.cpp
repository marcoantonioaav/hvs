#include "hvs_hdf5.h"
#include <hdf5.h>
#include <iostream>
#include <cmath>

namespace hvs {

bool read_hdf5_matrix(const std::string& filepath, 
                      const std::string& dataset_name, 
                      HDF5Matrix& out_matrix) {
    hid_t file_id = H5Fopen(filepath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "[HVS HDF5 Error] Failed to open HDF5 file: " << filepath << std::endl;
        return false;
    }

    // Ensure leading slash if missing
    std::string key = dataset_name;
    if (!key.empty() && key[0] != '/') {
        key = "/" + key;
    }

    hid_t dataset_id = H5Dopen2(file_id, key.c_str(), H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "[HVS HDF5 Error] Failed to open dataset: " << key << " in file " << filepath << std::endl;
        H5Fclose(file_id);
        return false;
    }

    hid_t dataspace_id = H5Dget_space(dataset_id);
    int rank = H5Sget_simple_extent_ndims(dataspace_id);
    if (rank != 2) {
        std::cerr << "[HVS HDF5 Error] Expected 2D dataset, but got rank " << rank << std::endl;
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }

    hsize_t dims[2];
    H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
    out_matrix.num_vectors = dims[0];
    out_matrix.dim = dims[1];

    out_matrix.data.resize(out_matrix.num_vectors * out_matrix.dim);

    // Read dataset into float vector
    herr_t status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, out_matrix.data.data());

    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);

    if (status < 0) {
        std::cerr << "[HVS HDF5 Error] Error reading data from dataset " << key << std::endl;
        return false;
    }

    std::cout << "[HVS HDF5] Successfully loaded " << out_matrix.num_vectors 
              << " vectors of dimension " << out_matrix.dim 
              << " from " << filepath << ":" << key << std::endl;
    return true;
}

void l2_normalize_matrix(HDF5Matrix& matrix) {
    for (size_t i = 0; i < matrix.num_vectors; ++i) {
        float* vec = &matrix.data[i * matrix.dim];
        float sum_sq = 0.0f;
        for (size_t d = 0; d < matrix.dim; ++d) {
            sum_sq += vec[d] * vec[d];
        }
        if (sum_sq > 0.0f) {
            float inv_norm = 1.0f / std::sqrt(sum_sq);
            for (size_t d = 0; d < matrix.dim; ++d) {
                vec[d] *= inv_norm;
            }
        }
    }
}

} // namespace hvs

