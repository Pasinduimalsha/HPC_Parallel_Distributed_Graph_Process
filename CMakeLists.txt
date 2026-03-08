cmake_minimum_required(VERSION 3.10)
project(HPC_PageRank)

set(CMAKE_C_STANDARD 99)

# Include directories
include_directories(include)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find MPI
find_package(MPI REQUIRED)

# Helper for math library (if needed on Linux/Unix, not needed on Windows but good practice)
if(NOT MSVC)
    link_libraries(m)
endif()

# Common files
set(CORE_SOURCES src/graph.c src/serial_pagerank.c)

# Serial implementation
add_executable(serial main/main_serial.c ${CORE_SOURCES})

# OpenMP implementation
add_executable(openmp main/main_openmp.c ${CORE_SOURCES} src/openmp_pagerank.c)
target_link_libraries(openmp PRIVATE OpenMP::OpenMP_C)

# MPI implementation
add_executable(mpi main/main_mpi.c ${CORE_SOURCES} src/mpi_pagerank.c)
target_link_libraries(mpi PRIVATE MPI::MPI_C)

# Tools
add_executable(generate_graph tools/generate_graph.c)

# Validation
add_executable(validation main/main_validation.c ${CORE_SOURCES} src/openmp_pagerank.c src/mpi_pagerank.c)
target_link_libraries(validation PRIVATE OpenMP::OpenMP_C MPI::MPI_C)

# Set output directory to bin to match Makefile
set_target_properties(serial openmp mpi generate_graph validation PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)
