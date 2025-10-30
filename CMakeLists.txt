cmake_minimum_required(VERSION 3.23)
project(BlockSparseTools CXX)

set(WARN_FLAGS -Wall)
# set(WARN_FLAGS_CUDA -Xcompiler=-Wall)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

# enable_language(CUDA)
# find_package(CUDAToolkit REQUIRED)

add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-g>)

add_executable(tensor_contraction
     src/main_tensor.cpp
     src/block_sparse_tensor.cpp
     src/tensor_layout.cpp
     src/validation.cpp
     src/tblis_bs_contraction.cpp
)

target_compile_options(tensor_contraction PRIVATE -O3)
target_compile_options(tensor_contraction PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${WARN_FLAGS}>
)

# Optional OpenMP for CPU packing
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(tensor_contraction PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(tensor_contraction PRIVATE USE_OPENMP)
    message(STATUS "OpenMP enabled for tensor_contraction")
endif()

# # Executable: tensor_generator
add_executable(tensor_generator
    tensor_generator.cpp
)
target_compile_options(tensor_generator PRIVATE -O3)
# Add warnings for C++ sources
target_compile_options(tensor_generator PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${WARN_FLAGS}>
)

add_executable(bs_flops
    src/bs_flops.cpp
)
target_compile_options(bs_flops PRIVATE -O3)
target_compile_options(bs_flops PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${WARN_FLAGS}>
)

# Add TBLIS executable
set(TBLIS_ROOT ${CMAKE_SOURCE_DIR}/tblis)
if(NOT EXISTS ${TBLIS_ROOT}/include)
    message(FATAL_ERROR "TBLIS headers not found at ${TBLIS_ROOT}/include")
endif()
target_include_directories(tensor_contraction PRIVATE
    ${TBLIS_ROOT}/include
)

find_library(TBLIS_LIBRARY tblis
    HINTS ${TBLIS_ROOT}/lib
    REQUIRED)
target_link_libraries(tensor_contraction PRIVATE ${TBLIS_LIBRARY})

set_source_files_properties(src/validation.cpp PROPERTIES LANGUAGE CXX)
