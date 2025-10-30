#pragma once

#include "block_sparse_tensor.h"

#include <cstdint>
#include <vector>

// Dense host tensor helper (column-major layout to match cuTENSOR default strides)
template <typename T> struct DenseHostTensor {
  std::vector<int64_t> extents; // per-mode extents in tensor order
  std::vector<T> data;          // column-major contiguous buffer
};

// Generate deterministic per-block values (for reproducible validation).
// The order of returned blocks matches iteration over T.coordinates.
template <typename T> std::vector<std::vector<T>> generate_block_values(const BlockSparseTensor &tensor, unsigned seed);

// Pack block-sparse data into a dense column-major buffer according to T's sectioning.
template <typename T>
DenseHostTensor<T> pack_blocks_to_dense(const BlockSparseTensor &tensor, const std::vector<std::vector<T>> &blocks);

template <typename T>
DenseHostTensor<T> run_dense_tblis(const std::vector<char> &Aidx, const std::vector<char> &Bidx,
                                      const std::vector<char> &Oidx, const DenseHostTensor<T> &A,
                                      const DenseHostTensor<T> &B, float &kernel_ms);

