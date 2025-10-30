#include "fast_rng.h"
#include "tensor_utils.h"
#include "validation.h"
#include "../tblis/src/tblis.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <chrono>

template <typename T>
std::vector<std::vector<T>> generate_block_values(const BlockSparseTensor &tensor, unsigned seed) {
  fast_rng::FastRng rng(seed);
  std::vector<std::vector<T>> out;
  out.reserve(tensor.coordinates.size());
  for (const auto &coord : tensor.coordinates) {
    size_t elems = 1;
    for (size_t i = 0; i < coord.size(); ++i) {
      char lbl = tensor.indices[i];
      elems *= static_cast<size_t>(tensor.block_sizes.at(lbl)[coord[i]]);
    }
    std::vector<T> b(elems);
    for (auto &v : b) v = rng.next_signed<T>();
    out.push_back(std::move(b));
  }
  return out;
}

// Compute strides for column-major layout: stride[k] = prod_{m<k} extents[m]
static std::vector<int64_t> column_major_strides(const std::vector<int64_t> &ext) {
  std::vector<int64_t> s(ext.size(), 1);
  for (size_t k = 1; k < ext.size(); ++k) s[k] = s[k - 1] * ext[k - 1];
  return s;
}

// Precompute per-mode section offsets (prefix sums of block sizes)
static std::map<char, std::vector<int64_t>> section_offsets(const BlockSparseTensor &tensor) {
  std::map<char, std::vector<int64_t>> offs;
  for (char lbl : tensor.indices) {
    const auto &sizes = tensor.block_sizes.at(lbl);
    std::vector<int64_t> o(sizes.size(), 0);
    int64_t acc = 0;
    for (size_t i = 0; i < sizes.size(); ++i) {
      o[i] = acc;
      acc += sizes[i];
    }
    offs[lbl] = std::move(o);
  }
  return offs;
}

template <typename T>
DenseHostTensor<T> pack_blocks_to_dense(const BlockSparseTensor &tensor, const std::vector<std::vector<T>> &blocks) {
  if (blocks.size() != tensor.coordinates.size()) throw std::invalid_argument("blocks size mismatch");
  DenseHostTensor<T> H;
  const auto dims = tensor.get_total_dimensions();
  H.extents.assign(dims.begin(), dims.end());
  int64_t total = 1;
  for (auto d : H.extents) total *= d;
  H.data.assign(total, static_cast<T>(0));
  const auto strides = column_major_strides(H.extents);
  const auto offs = section_offsets(tensor);

  size_t bi = 0;
  for (const auto &coord : tensor.coordinates) {
    const size_t D = coord.size();
    // Fast path requires D <= MAX_TENSOR_RANK (guarded in main)
    int64_t bext[MAX_TENSOR_RANK];
    int64_t base[MAX_TENSOR_RANK];
    int64_t gstep[MAX_TENSOR_RANK];
    for (size_t m = 0; m < D; ++m) {
      char lbl = tensor.indices[m];
      int sec = coord[m];
      bext[m] = tensor.block_sizes.at(lbl)[sec];
      base[m] = offs.at(lbl)[sec];
      gstep[m] = strides[m];
    }
    int64_t belems = 1;
    for (size_t m = 0; m < D; ++m) belems *= bext[m];
    int64_t lin = 0;
    for (size_t m = 0; m < D; ++m) lin += base[m] * gstep[m];
    int64_t local[MAX_TENSOR_RANK] = {0};
    for (int64_t t = 0; t < belems; ++t) {
      H.data[lin] = blocks[bi][t];
      for (size_t m = 0; m < D; ++m) {
        if (++local[m] < bext[m]) {
          lin += gstep[m];
          break;
        } else {
          const int64_t reset = bext[m] - 1;
          local[m] = 0;
          lin -= gstep[m] * reset;
        }
      }
    }
    ++bi;
  }
  return H;
}

template <typename T>
DenseHostTensor<T> run_dense_tblis(const std::vector<char> &Aidx, const std::vector<char> &Bidx,
                                   const std::vector<char> &Oidx, const DenseHostTensor<T> &A,
                                   const DenseHostTensor<T> &B, float &kernel_ms) {
  DenseHostTensor<T> O;
  O.extents = {};
  {
    std::map<char, int64_t> edict;
    for (size_t i = 0; i < Aidx.size(); ++i) edict[Aidx[i]] = A.extents[i];
    for (size_t i = 0; i < Bidx.size(); ++i) edict[Bidx[i]] = B.extents[i];
    for (char c : Oidx) O.extents.push_back(edict.at(c));
  }
  
  std::vector<tblis::len_type> dimA, dimB, dimO;
  for (auto v : A.extents) {
    dimA.push_back(static_cast<tblis::len_type>(v));
  }
  for (auto v : B.extents) {
    dimB.push_back(static_cast<tblis::len_type>(v));
  }
  auto totalO = 1;
  for (auto v : O.extents) {
    dimO.push_back(static_cast<tblis::len_type>(v));
    totalO *= v;
  }
  // Create row-major ordered data for printing
  // Convert A from column-major to row-major
  std::vector<T> row_major_A(A.data.size());
  std::vector<int64_t> row_strides_A(A.extents.size());
  row_strides_A[A.extents.size()-1] = 1;
  for (int i = A.extents.size()-2; i >= 0; --i) {
    row_strides_A[i] = row_strides_A[i+1] * A.extents[i+1];
  }

  for (size_t i = 0; i < A.data.size(); ++i) {
    size_t col_idx = i;
    size_t row_idx = 0;
    for (size_t dim = 0; dim < A.extents.size(); ++dim) {
      size_t coord = col_idx % A.extents[dim];
      col_idx /= A.extents[dim];
      row_idx += coord * row_strides_A[dim];
    }
    row_major_A[row_idx] = A.data[i];
  }

  // Convert B from column-major to row-major
  std::vector<T> row_major_B(B.data.size());
  std::vector<int64_t> row_strides_B(B.extents.size());
  row_strides_B[B.extents.size()-1] = 1;
  for (int i = B.extents.size()-2; i >= 0; --i) {
    row_strides_B[i] = row_strides_B[i+1] * B.extents[i+1];
  }

  for (size_t i = 0; i < B.data.size(); ++i) {
    size_t col_idx = i;
    size_t row_idx = 0;
    for (size_t dim = 0; dim < B.extents.size(); ++dim) {
      size_t coord = col_idx % B.extents[dim];
      col_idx /= B.extents[dim];
      row_idx += coord * row_strides_B[dim];
    }
    row_major_B[row_idx] = B.data[i];
  }

  std::vector<T> dataA, dataB, dataO;
  for(size_t i = 0; i < row_major_A.size(); i++) {
    dataA.push_back(static_cast<T>(row_major_A[i]));
  }
  for(size_t i = 0; i < row_major_B.size(); i++) {
    dataB.push_back(static_cast<T>(row_major_B[i]));
  }
  for (size_t i = 0; i < totalO; i++)
  {
    dataO.push_back(static_cast<T>(0));
  }
  
  MArray::varray_view<T> At(dimA, dataA.data());
  MArray::varray_view<T> Bt(dimB, dataB.data());
  MArray::varray_view<T> Ot(dimO, dataO.data());
  tblis::label_vector idx_A, idx_B, idx_O;
  for(auto c : Aidx) {
    idx_A.push_back(c);
  }
  for(auto c : Bidx) {
    idx_B.push_back(c);
  }
  for(auto c : Oidx) {
    idx_O.push_back(c);
  }
  auto start = std::chrono::high_resolution_clock::now();
  tblis::mult<T>(static_cast<T>(1), At, idx_A.data(), Bt, idx_B.data(), static_cast<T>(0), Ot, idx_O.data());
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  kernel_ms = static_cast<float>(elapsed.count() * 1000.0);
  // Convert dataO (row-major) -> column-major
std::vector<T> col_major_O(dataO.size());

// --- row-major strides (last dim is fastest) ---
std::vector<size_t> row_strides_O(O.extents.size());
row_strides_O.back() = 1;
for (std::ptrdiff_t d = (std::ptrdiff_t)O.extents.size() - 2; d >= 0; --d) {
  row_strides_O[d] = row_strides_O[d + 1] * static_cast<size_t>(O.extents[d + 1]);
}

// --- column-major strides (first dim is fastest) ---
std::vector<size_t> col_strides_O(O.extents.size());
col_strides_O[0] = 1;
for (size_t d = 1; d < O.extents.size(); ++d) {
  col_strides_O[d] = col_strides_O[d - 1] * static_cast<size_t>(O.extents[d - 1]);
}

// --- map row-major linear index -> column-major linear index ---
for (size_t i = 0; i < dataO.size(); ++i) {
  size_t row_idx = i;     // row-major linear index
  size_t col_idx = 0;     // column-major linear index (to compute)

  for (size_t d = 0; d < O.extents.size(); ++d) {
    const size_t coord = row_idx / row_strides_O[d];
    row_idx %= row_strides_O[d];
    col_idx += coord * col_strides_O[d];
  }
  col_major_O[col_idx] = dataO[i];
}

  O.data = std::vector<T>(col_major_O.begin(), col_major_O.end());
  return O;
}



// Explicit template instantiations for float and double
template DenseHostTensor<float> pack_blocks_to_dense<float>(const BlockSparseTensor &tensor, const std::vector<std::vector<float>> &blocks);
template DenseHostTensor<double> pack_blocks_to_dense<double>(const BlockSparseTensor &tensor, const std::vector<std::vector<double>> &blocks);
template DenseHostTensor<float> run_dense_tblis<float>(const std::vector<char> &Aidx, const std::vector<char> &Bidx,
                                   const std::vector<char> &Oidx, const DenseHostTensor<float> &A,
                                   const DenseHostTensor<float> &B, float &kernel_ms);
template DenseHostTensor<double> run_dense_tblis<double>(const std::vector<char> &Aidx, const std::vector<char> &Bidx,
                                   const std::vector<char> &Oidx, const DenseHostTensor<double> &A,
                                   const DenseHostTensor<double> &B, float &kernel_ms);
template std::vector<std::vector<float>> generate_block_values<float>(const BlockSparseTensor &tensor, unsigned seed);
template std::vector<std::vector<double>> generate_block_values<double>(const BlockSparseTensor &tensor, unsigned seed);