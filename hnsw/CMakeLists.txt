cmake_minimum_required (VERSION 3.10)
project (hvs_benchmark)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HDF5_INCLUDE_DIRS}
    ${MPI_CXX_INCLUDE_PATH}
    ${MPI_C_INCLUDE_PATH}
)

set(HVS_LIB_SOURCES
    hvs_hdf5.cpp
    hvs_index.cpp
)

add_library(hvs_static STATIC ${HVS_LIB_SOURCES})
target_link_libraries(hvs_static ${HDF5_LIBRARIES} ${MPI_CXX_LIBRARIES})

add_executable(main main.cpp ${HVS_LIB_SOURCES})
set(CMAKE_CXX_FLAGS "-Ofast -DNDEBUG -march=native -fpic -w -fopenmp")

target_link_libraries(main ${HDF5_LIBRARIES} ${MPI_CXX_LIBRARIES})
