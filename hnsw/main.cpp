#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <cstdlib>
#include "hvs_hdf5.h"
#include "hvs_index.h"

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " <dataset.h5> <query.h5> [metric=l2|cosine] [k=10] [levels=1] [ef_search=1000] [output.txt]\n"
              << "Example:\n"
              << "  " << prog_name << " dataset.h5 query.h5 l2 10 1 1000 results.txt\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    std::string dataset_file = argv[1];
    std::string query_file = argv[2];
    std::string metric_str = (argc > 3) ? argv[3] : "l2";
    size_t k = (argc > 4) ? std::stoul(argv[4]) : 10;
    int levels = (argc > 5) ? std::stoi(argv[5]) : 1;
    int ef_search = (argc > 6) ? std::stoi(argv[6]) : 1000;
    std::string output_file = (argc > 7) ? argv[7] : "";

    hvs::MetricType metric = hvs::MetricType::L2;
    if (metric_str == "cosine" || metric_str == "COSINE") {
        metric = hvs::MetricType::COSINE;
    }

    // Load dataset and queries using default HDF5 dataset key "/embeddings"
    hvs::HDF5Matrix dataset_mat, query_mat;
    std::string key = "/embeddings";

    std::cout << "[HVS Benchmark] Loading dataset from: " << dataset_file << " (" << key << ")\n";
    if (!hvs::read_hdf5_matrix(dataset_file, key, dataset_mat)) {
        std::cerr << "Failed to read dataset from " << dataset_file << std::endl;
        return 1;
    }

    std::cout << "[HVS Benchmark] Loading queries from: " << query_file << " (" << key << ")\n";
    if (!hvs::read_hdf5_matrix(query_file, key, query_mat)) {
        std::cerr << "Failed to read queries from " << query_file << std::endl;
        return 1;
    }

    // Build HVS Index
    hvs::HVSIndex index(levels, 0.5f, metric);
    std::cout << "[HVS Benchmark] Building HVS index...\n";
    auto start_build = std::chrono::high_resolution_clock::now();
    if (!index.build(dataset_mat)) {
        std::cerr << "Failed to build HVS index." << std::endl;
        return 1;
    }
    auto end_build = std::chrono::high_resolution_clock::now();
    double build_time = std::chrono::duration<double>(end_build - start_build).count();
    std::cout << "[HVS Benchmark] Index build complete in " << build_time << " seconds.\n";

    // Run batch search
    std::cout << "[HVS Benchmark] Executing queries (k=" << k << ", ef_search=" << ef_search << ")...\n";
    auto start_search = std::chrono::high_resolution_clock::now();
    std::vector<hvs::QueryResult> results = index.search_batch(query_mat, k, ef_search);
    auto end_search = std::chrono::high_resolution_clock::now();

    double search_time = std::chrono::duration<double>(end_search - start_search).count();
    size_t total_dist_comps = 0;
    for (const auto& res : results) {
        total_dist_comps += res.distance_computations;
    }

    double avg_dist_comps = (query_mat.num_vectors > 0) ? (double)total_dist_comps / query_mat.num_vectors : 0.0;
    double qps = (search_time > 0.0) ? (query_mat.num_vectors / search_time) : 0.0;

    std::cout << "\n================ HVS Benchmark Results ================\n";
    std::cout << "Queries Processed    : " << query_mat.num_vectors << "\n";
    std::cout << "Total Search Time    : " << search_time << " s\n";
    std::cout << "QPS (Queries/sec)    : " << qps << "\n";
    std::cout << "Total Dist Comps     : " << total_dist_comps << "\n";
    std::cout << "Avg Dist Comps/Query : " << avg_dist_comps << "\n";
    std::cout << "========================================================\n\n";

    // Save or output sample k-NN results
    std::ostream* out_stream = &std::cout;
    std::ofstream fout;
    if (!output_file.empty()) {
        fout.open(output_file);
        if (fout.is_open()) {
            out_stream = &fout;
            std::cout << "[HVS Benchmark] Saving k-NN results to " << output_file << std::endl;
        }
    }

    for (size_t q = 0; q < results.size(); ++q) {
        *out_stream << "Query " << q << " (dist_comps=" << results[q].distance_computations << "):\n";
        for (size_t i = 0; i < results[q].neighbor_ids.size(); ++i) {
            *out_stream << "  Neighbor " << i << ": ID=" << results[q].neighbor_ids[i]
                        << ", Dist=" << results[q].distances[i] << "\n";
        }
    }

    if (fout.is_open()) fout.close();

    return 0;
}
