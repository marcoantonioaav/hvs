# Standalone HVS Index for ANNS Seed Selection Benchmarking

This repository is a refactored implementation of **HVS (Hierarchical Graph Structure Based on Voronoi Diagrams)**, decoupled from HNSW graph dependencies to serve as a standalone seed selection and vector index benchmark engine.

Original Paper: *"HVS: Hierarchical Graph Structure Based on Voronoi Diagrams for Solving Approximate Nearest Neighbor Search"* ([VLDB 2021](https://www.vldb.org/pvldb/vol15/p246-lu.pdf)).

---

## Features

- **Decoupled HVS Core**: The HVS Voronoi tree navigation logic (`hvs::HVSIndex`) operates independently of HNSW graph edge structures.
- **HDF5 `.h5` Ingestion**: Directly reads base dataset vectors and query vectors from HDF5 files using the `/embeddings` dataset key.
- **Distance Computation Tracking**: Tracks exact metric evaluations per query and reports average distance computations alongside Query-per-Second (QPS) throughput.
- **Top-$k$ Neighbor IDs & Distances**: Outputs calculated neighbor indices and distances directly without needing ground truth (`.gt`) files.
- **Dual Distance Metrics**: Supports both Euclidean distance (`l2`) and `cosine` distance.

---

## Prerequisites

- **Compiler**: GCC with C++14 standard support and OpenMP (`-fopenmp`)
- **Build System**: CMake (v3.10+)
- **Libraries**:
  - `libhdf5-dev` (HDF5 development libraries)
  - `openmpi-bin` / `libhdf5-openmpi-dev` (if using parallel HDF5)

On Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install build-essential cmake libhdf5-dev
```

---

## How to Compile

Build the static library `libhvs_static.a` and the benchmark CLI binary `main`:

```bash
cd hnsw/
mkdir -p build && cd build
cmake ..
make
```

---

## Running the Benchmark CLI

The executable accepts the dataset `.h5` file, query `.h5` file, distance metric, top-$k$, hierarchy level count, search beam width ($ef_{search}$), and optional output file path.

### Command Syntax

```bash
./hnsw/build/main <dataset.h5> <query.h5> [metric] [k] [levels] [ef_search] [output.txt]
```

### Parameters

- `<dataset.h5>`: Path to base dataset HDF5 file (must contain dataset key `/embeddings`).
- `<query.h5>`: Path to query dataset HDF5 file (must contain dataset key `/embeddings`).
- `[metric]`: Distance metric type (`l2` or `cosine`, default: `l2`).
- `[k]`: Top-$k$ nearest neighbors to retrieve (default: `10`).
- `[levels]`: Number of HVS hierarchy levels $T$ (default: `1`).
- `[ef_search]`: Search beam width / candidate pool size (default: `1000`).
- `[output.txt]`: Optional file path to save $k$-NN neighbor IDs and distances.

### Usage Example

```bash
# Run with L2 distance
./hnsw/build/main dataset.h5 query.h5 l2 10 1 1000 output_l2.txt

# Run with Cosine distance
./hnsw/build/main dataset.h5 query.h5 cosine 10 1 1000 output_cosine.txt
```

---

## C++ API Usage

You can also link against `libhvs_static.a` and use the C++ API directly:

```cpp
#include "hvs_hdf5.h"
#include "hvs_index.h"

// Load dataset and queries from HDF5 under /embeddings
hvs::HDF5Matrix dataset, queries;
hvs::read_hdf5_matrix("dataset.h5", "/embeddings", dataset);
hvs::read_hdf5_matrix("query.h5", "/embeddings", queries);

// Build HVS Index
hvs::HVSIndex index(1 /* levels */, 0.5f /* delta */, hvs::MetricType::L2);
index.build(dataset);

// Batch Search
std::vector<hvs::QueryResult> results = index.search_batch(queries, 10 /* k */, 1000 /* ef_search */);

for (size_t q = 0; q < results.size(); ++q) {
    std::cout << "Query " << q << " evaluated " << results[q].distance_computations << " distances.\n";
    for (size_t i = 0; i < results[q].neighbor_ids.size(); ++i) {
        std::cout << "  ID: " << results[q].neighbor_ids[i] << ", Dist: " << results[q].distances[i] << "\n";
    }
}
```
