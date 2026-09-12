#ifndef HVS_HDF5_H
#define HVS_HDF5_H

#include <string>
#include <vector>
#include <cstddef>

namespace hvs {

struct HDF5Matrix {
    std::vector<float> data;
    size_t num_vectors = 0;
    size_t dim = 0;
};

// Reads a 2D float dataset from an HDF5 file (defaulting to dataset_name = "/embeddings")
bool read_hdf5_matrix(const std::string& filepath, 
                      const std::string& dataset_name, 
                      HDF5Matrix& out_matrix);

// Normalizes vectors in-place for Cosine metric processing (L2 norm)
void l2_normalize_matrix(HDF5Matrix& matrix);

} // namespace hvs

#endif // HVS_HDF5_H

