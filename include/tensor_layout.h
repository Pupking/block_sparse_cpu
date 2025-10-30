#pragma once

#include "block_sparse_tensor.h"

#include <set>
#include <vector>

std::set<std::vector<int>>
build_output_coordinates(const BlockSparseTensor &A, const BlockSparseTensor &B, const EinsumOperation &op);

long double compute_blocksparse_actual_flops(const BlockSparseTensor &A, const BlockSparseTensor &B,
                                             const EinsumOperation &op);
