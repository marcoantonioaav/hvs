#include "hvs_index.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <queue>
#include <cmath>
#include <cstring>

namespace hvs {

static const int OFF = 1;
static const int CEN = 16; // centroids
static const int L_VAL = 16;
static const int FAN = 32;

HVSIndex::HVSIndex(int levels, float delta, MetricType metric)
    : max_level_(levels), delta_(delta), metric_(metric) {
    if (max_level_ < 1) max_level_ = 1;
}

HVSIndex::~HVSIndex() {}

float HVSIndex::compute_distance(const float* a, const float* b, size_t dim, size_t& dist_counter) const {
    dist_counter++;
    float dist = 0.0f;
    if (metric_ == MetricType::L2) {
        for (size_t i = 0; i < dim; ++i) {
            float diff = a[i] - b[i];
            dist += diff * diff;
        }
    } else { // COSINE distance (assumes normalized vectors)
        float dot = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            dot += a[i] * b[i];
        }
        dist = 1.0f - dot;
        if (dist < 0.0f) dist = 0.0f;
    }
    return dist;
}

bool HVSIndex::build(const HDF5Matrix& matrix) {
    num_vectors_ = matrix.num_vectors;
    dim_ = matrix.dim;

    if (num_vectors_ == 0 || dim_ == 0) return false;

    // Copy dataset and normalize if Cosine metric
    dataset_ = matrix.data;
    if (metric_ == MetricType::COSINE) {
        for (size_t i = 0; i < num_vectors_; ++i) {
            float* vec = &dataset_[i * dim_];
            float sum_sq = 0.0f;
            for (size_t d = 0; d < dim_; ++d) sum_sq += vec[d] * vec[d];
            if (sum_sq > 0.0f) {
                float inv_norm = 1.0f / std::sqrt(sum_sq);
                for (size_t d = 0; d < dim_; ++d) vec[d] *= inv_norm;
            }
        }
    }

    int max_num = static_cast<int>(std::pow(2, max_level_ + OFF));
    int remainder = dim_ % max_num;
    int ratio = dim_ / max_num;
    dim_padded_ = (remainder == 0) ? dim_ : (ratio + 1) * max_num;

    length_.resize(max_level_);
    dim_sub_.resize(max_level_);
    for (int i = 0; i < max_level_; ++i) {
        length_[i] = static_cast<int>(std::pow(2, max_level_ - i + OFF));
        dim_sub_[i] = dim_padded_ / length_[i];
    }

    // Initialize Identity Rotation Matrix by default (or compute PCA)
    R_.assign(dim_padded_ * dim_padded_, 0.0f);
    for (size_t i = 0; i < dim_padded_; ++i) {
        R_[i * dim_padded_ + i] = 1.0f;
    }

    // Allocate start_book_ table (flat 1D vector)
    size_t tol = CEN * CEN * CEN * CEN;
    start_book_.assign(tol * FAN, 0);

    // Create level mapping arrays
    init_obj_.assign(max_level_, std::vector<int>(num_vectors_, 0));
    for (int lvl = 0; lvl < max_level_; ++lvl) {
        for (size_t i = 0; i < num_vectors_; ++i) {
            init_obj_[lvl][i] = static_cast<int>(i);
        }
    }

    std::cout << "[HVS Index] Built structure: " << num_vectors_ << " vectors, dim=" << dim_ 
              << ", levels=" << max_level_ << ", metric=" << (metric_ == MetricType::L2 ? "L2" : "Cosine") 
              << std::endl;
    return true;
}

bool HVSIndex::build(const std::vector<std::vector<float>>& dataset) {
    if (dataset.empty() || dataset[0].empty()) return false;
    num_vectors_ = dataset.size();
    dim_ = dataset[0].size();

    dataset_.resize(num_vectors_ * dim_);
    for (size_t i = 0; i < num_vectors_; ++i) {
        std::copy(dataset[i].begin(), dataset[i].end(), dataset_.begin() + i * dim_);
    }

    if (metric_ == MetricType::COSINE) {
        for (size_t i = 0; i < num_vectors_; ++i) {
            float* vec = &dataset_[i * dim_];
            float sum_sq = 0.0f;
            for (size_t d = 0; d < dim_; ++d) sum_sq += vec[d] * vec[d];
            if (sum_sq > 0.0f) {
                float inv_norm = 1.0f / std::sqrt(sum_sq);
                for (size_t d = 0; d < dim_; ++d) vec[d] *= inv_norm;
            }
        }
    }

    int max_num = static_cast<int>(std::pow(2, max_level_ + OFF));
    int remainder = dim_ % max_num;
    int ratio = dim_ / max_num;
    dim_padded_ = (remainder == 0) ? dim_ : (ratio + 1) * max_num;

    length_.resize(max_level_);
    dim_sub_.resize(max_level_);
    for (int i = 0; i < max_level_; ++i) {
        length_[i] = static_cast<int>(std::pow(2, max_level_ - i + OFF));
        dim_sub_[i] = dim_padded_ / length_[i];
    }

    R_.assign(dim_padded_ * dim_padded_, 0.0f);
    for (size_t i = 0; i < dim_padded_; ++i) {
        R_[i * dim_padded_ + i] = 1.0f;
    }

    size_t tol = CEN * CEN * CEN * CEN;
    start_book_.assign(tol * FAN, 0);

    init_obj_.assign(max_level_, std::vector<int>(num_vectors_, 0));
    for (int lvl = 0; lvl < max_level_; ++lvl) {
        for (size_t i = 0; i < num_vectors_; ++i) {
            init_obj_[lvl][i] = static_cast<int>(i);
        }
    }

    std::cout << "[HVS Index] Built structure: " << num_vectors_ << " vectors, dim=" << dim_ 
              << ", levels=" << max_level_ << ", metric=" << (metric_ == MetricType::L2 ? "L2" : "Cosine") 
              << std::endl;
    return true;
}

bool HVSIndex::save_index(const std::string& index_prefix) const {
    std::ofstream out(index_prefix + ".hvs", std::ios::binary);
    if (!out.is_open()) return false;

    out.write(reinterpret_cast<const char*>(&num_vectors_), sizeof(num_vectors_));
    out.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
    out.write(reinterpret_cast<const char*>(&dim_padded_), sizeof(dim_padded_));
    out.write(reinterpret_cast<const char*>(&max_level_), sizeof(max_level_));
    int metric_val = static_cast<int>(metric_);
    out.write(reinterpret_cast<const char*>(&metric_val), sizeof(metric_val));

    out.close();
    return true;
}

bool HVSIndex::load_index(const std::string& index_prefix, const HDF5Matrix& matrix) {
    std::ifstream in(index_prefix + ".hvs", std::ios::binary);
    if (!in.is_open()) {
        std::cout << "[HVS Index] Index file not found, building from matrix." << std::endl;
        return build(matrix);
    }

    in.read(reinterpret_cast<char*>(&num_vectors_), sizeof(num_vectors_));
    in.read(reinterpret_cast<char*>(&dim_), sizeof(dim_));
    in.read(reinterpret_cast<char*>(&dim_padded_), sizeof(dim_padded_));
    in.read(reinterpret_cast<char*>(&max_level_), sizeof(max_level_));
    int metric_val;
    in.read(reinterpret_cast<char*>(&metric_val), sizeof(metric_val));
    metric_ = static_cast<MetricType>(metric_val);
    in.close();

    return build(matrix);
}

QueryResult HVSIndex::search_query(const float* query_vec, size_t k, int ef_search) const {
    QueryResult result;
    result.distance_computations = 0;

    if (num_vectors_ == 0 || k == 0) return result;

    // Standardize query vector (normalize if Cosine metric)
    std::vector<float> q_norm(dim_);
    if (metric_ == MetricType::COSINE) {
        float sum_sq = 0.0f;
        for (size_t d = 0; d < dim_; ++d) sum_sq += query_vec[d] * query_vec[d];
        float inv_norm = (sum_sq > 0.0f) ? (1.0f / std::sqrt(sum_sq)) : 1.0f;
        for (size_t d = 0; d < dim_; ++d) q_norm[d] = query_vec[d] * inv_norm;
    } else {
        std::copy(query_vec, query_vec + dim_, q_norm.begin());
    }

    // Evaluate candidates using distance calculation on dataset
    std::priority_queue<std::pair<float, size_t>> pq; // max-heap for top-k

    size_t search_budget = std::min(static_cast<size_t>(ef_search), num_vectors_);
    for (size_t i = 0; i < search_budget; ++i) {
        const float* base_vec = &dataset_[i * dim_];
        float dist = compute_distance(q_norm.data(), base_vec, dim_, result.distance_computations);

        if (pq.size() < k) {
            pq.push({dist, i});
        } else if (dist < pq.top().first) {
            pq.pop();
            pq.push({dist, i});
        }
    }

    // Extract sorted top-k results
    std::vector<std::pair<float, size_t>> sorted_res;
    while (!pq.empty()) {
        sorted_res.push_back(pq.top());
        pq.pop();
    }
    std::reverse(sorted_res.begin(), sorted_res.end());

    for (const auto& item : sorted_res) {
        result.distances.push_back(item.first);
        result.neighbor_ids.push_back(item.second);
    }

    return result;
}

std::vector<QueryResult> HVSIndex::search_batch(const HDF5Matrix& query_matrix, size_t k, int ef_search) const {
    std::vector<QueryResult> results(query_matrix.num_vectors);
    #pragma omp parallel for
    for (size_t i = 0; i < query_matrix.num_vectors; ++i) {
        const float* q_vec = &query_matrix.data[i * query_matrix.dim];
        results[i] = search_query(q_vec, k, ef_search);
    }
    return results;
}

} // namespace hvs
