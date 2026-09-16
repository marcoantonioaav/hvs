#ifndef HVS_INDEX_H
#define HVS_INDEX_H

#include <vector>
#include <string>
#include <cstddef>
#include <utility>
#include <cmath>
#include "hvs_hdf5.h"

namespace hvs {

enum class MetricType {
    L2,
    COSINE
};

struct QueryResult {
    std::vector<size_t> neighbor_ids;
    std::vector<float> distances;
    size_t distance_computations = 0;
};

class HVSIndex {
public:
    HVSIndex(int levels = 1, float delta = 0.5f, MetricType metric = MetricType::L2);
    ~HVSIndex();

    // Build the HVS index from an input dataset matrix
    // Build the HVS index from an input dataset matrix or 2D vector
    bool build(const HDF5Matrix& matrix);
    bool build(const std::vector<std::vector<float>>& dataset);

    // Save and load index files
    bool save_index(const std::string& index_prefix) const;
    bool load_index(const std::string& index_prefix, const HDF5Matrix& matrix);

    // Perform HVS search for a single query vector
    QueryResult search_query(const float* query_vec, size_t k, int ef_search = 1000) const;

    // Perform HVS search for all queries in a query matrix
    std::vector<QueryResult> search_batch(const HDF5Matrix& query_matrix, size_t k, int ef_search = 1000) const;

    size_t get_num_vectors() const { return num_vectors_; }
    size_t get_dimension() const { return dim_; }
    MetricType get_metric() const { return metric_; }

private:
    int max_level_;
    float delta_;
    MetricType metric_;

    size_t num_vectors_ = 0;
    size_t dim_ = 0;
    size_t dim_padded_ = 0;

    // Stored base dataset (normalized if Cosine)
    std::vector<float> dataset_;

    // HVS codebooks and rotation
    std::vector<float> R_; // Rotation matrix (dim_padded_ * dim_padded_)
    
    // Level parameter arrays
    std::vector<int> length_;
    std::vector<int> dim_sub_;
    std::vector<int> count_obj_;

    // Quantizer centroids and structures
    // Codebooks per level: [level][subspace_idx][centroid_idx][sub_dim]
    std::vector<std::vector<std::vector<std::vector<float>>>> quantizer_;

    // Start book for initial candidate selection
    std::vector<std::vector<unsigned int>> start_book_;
    // Start book for initial candidate selection (flat 1D vector)
    std::vector<unsigned int> start_book_;

    // Object mappings across levels
    std::vector<std::vector<int>> init_obj_;
    
    // Object quantized codes
    std::vector<std::vector<std::vector<unsigned char>>> obj_codes_;

    // Helper functions
    float compute_distance(const float* a, const float* b, size_t dim, size_t& dist_counter) const;
};

} // namespace hvs

#endif // HVS_INDEX_H
