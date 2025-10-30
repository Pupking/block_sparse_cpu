#pragma once

#include "block_sparse_tensor.h"
#include "../tblis/src/tblis.h"
#include <vector>

// Result carrying numeric blocks for the output (host memory).
template <typename T> struct BlockSparseNumericResult {
  BlockSparseTensor tensor;           // structure of D
  std::vector<std::vector<T>> blocks; // one buffer per D block, in D.coordinates iteration order
  float kernel_ms = 0.0f;             // measured kernel time
  long double flops = 0.0L;           // actual FLOPs executed (sum over all GEMMs)
};


template <typename T>
BlockSparseNumericResult<T> contract_block_sparse_and_accumulate_tblis(const BlockSparseTensor &A, const BlockSparseTensor &B,
                                                                  const EinsumOperation &op,
                                                                  const std::vector<std::vector<T>> &A_blocks_host,
                                                                  const std::vector<std::vector<T>> &B_blocks_host);

