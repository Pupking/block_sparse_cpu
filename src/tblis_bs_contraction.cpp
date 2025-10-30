#include "tblis_bs_contraction.h"
#include "tensor_layout.h"
#include "tensor_utils.h"
#include <stdexcept>

#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cutensor.h>
#include <iostream>
#include <random>
#include <type_traits>
#include <chrono>

namespace {

// --- Algorithmic helpers ----------------------------------------------------

struct LabelGroups {
  std::vector<char> Arow; // free dims from A (rows)
  std::vector<char> Brow; // free dims from B (cols)
  std::vector<char> K;    // contraction dims (K)
};

static LabelGroups derive_label_groups(const BlockSparseTensor &A, const BlockSparseTensor &B,
                                       const EinsumOperation &op) {
  LabelGroups g;
  for (char c : op.output_indices) {
    bool inA = std::find(A.indices.begin(), A.indices.end(), c) != A.indices.end();
    if (inA)
      g.Arow.push_back(c);
    else
      g.Brow.push_back(c);
  }
  g.K = op.contraction_indices;
  return g;
}

static BlockSparseTensor make_output_tensor(const BlockSparseTensor &A, const BlockSparseTensor &B,
                                            const EinsumOperation &op) {
  BlockSparseTensor D;
  D.name = "Output";
  D.indices = op.output_indices;
  for (char lbl : D.indices) {
    const auto &src = (std::find(A.indices.begin(), A.indices.end(), lbl) != A.indices.end()) ? A : B;
    D.num_blocks.push_back((int)src.block_sizes.at(lbl).size());
    D.block_sizes[lbl] = src.block_sizes.at(lbl);
  }
  D.coordinates = build_output_coordinates(A, B, op);
  return D;
}

static std::vector<int> build_template_for_free_labels(const std::vector<char> &tensor_labels,
                                                       const std::vector<char> &free_labels, const BlockSparseTensor &D,
                                                       const std::vector<int> &co) {
  std::vector<int> templ(tensor_labels.size(), -1);
  for (char lbl : free_labels) {
    int it = -1;
    for (int i = 0; i < (int)tensor_labels.size(); ++i)
      if (tensor_labels[i] == lbl) {
        it = i;
        break;
      }
    int io = -1;
    for (int i = 0; i < (int)D.indices.size(); ++i)
      if (D.indices[i] == lbl) {
        io = i;
        break;
      }
    templ[it] = co[io];
  }
  return templ;
}

static std::set<std::vector<int>> enumerate_k_tuples(const BlockSparseTensor &A, const BlockSparseTensor &B,
                                                     const std::vector<int> &templA, const std::vector<int> &templB,
                                                     const std::vector<char> &Ak, const std::vector<char> &Bk) {
  std::set<std::vector<int>> k_tuples;
  // From A: collect tuples compatible with templA
  for (const auto &ca : A.coordinates) {
    bool match = true;
    for (size_t i = 0; i < A.indices.size(); ++i)
      if (templA[i] != -1 && ca[i] != templA[i]) {
        match = false;
        break;
      }
    if (!match) continue;
    std::vector<int> kt;
    kt.reserve(Ak.size());
    for (char k : Ak) {
      int ia = -1;
      for (int j = 0; j < (int)A.indices.size(); ++j)
        if (A.indices[j] == k) {
          ia = j;
          break;
        }
      kt.push_back(ca[ia]);
    }
    k_tuples.insert(std::move(kt));
  }
  // Intersect with B compatible tuples
  std::set<std::vector<int>> k_keep;
  for (const auto &cb : B.coordinates) {
    bool match = true;
    for (size_t i = 0; i < B.indices.size(); ++i)
      if (templB[i] != -1 && cb[i] != templB[i]) {
        match = false;
        break;
      }
    if (!match) continue;
    std::vector<int> kt;
    kt.reserve(Bk.size());
    for (char k : Bk) {
      int ib = -1;
      for (int j = 0; j < (int)B.indices.size(); ++j)
        if (B.indices[j] == k) {
          ib = j;
          break;
        }
      kt.push_back(cb[ib]);
    }
    if (k_tuples.count(kt)) k_keep.insert(std::move(kt));
  }
  k_tuples.swap(k_keep);
  return k_tuples;
}

static int64_t k_extent_product(const BlockSparseTensor &T, const std::vector<char> &klabels,
                                const std::vector<int> &ktuple) {
  int64_t p = 1;
  for (size_t idx = 0; idx < klabels.size(); ++idx) {
    char lbl = klabels[idx];
    int sec = ktuple[idx];
    p *= T.block_sizes.at(lbl)[sec];
  }
  return p;
}

// --- Existing local utilities (light cleanup) --------------------------------

// Helper to map coordinate->index in blocks vector (iteration order of set)
template <typename Elem>
static std::map<std::vector<int>, size_t> coord_index_map(const BlockSparseTensor &tensor,
                                                          const std::vector<std::vector<Elem>> &blocks) {
  if (blocks.size() != tensor.coordinates.size()) throw std::invalid_argument("blocks size mismatch");
  std::map<std::vector<int>, size_t> m;
  size_t i = 0;
  for (const auto &c : tensor.coordinates) { m[c] = i++; }
  return m;
}

// Validate that output indices are disjoint between A and B (no batch dims)
static void ensure_disjoint_output_labels(const BlockSparseTensor &A, const BlockSparseTensor &B,
                                          const EinsumOperation &op) {
  std::set<char> setA(A.indices.begin(), A.indices.end());
  std::set<char> setB(B.indices.begin(), B.indices.end());
  std::set<char> out(op.output_indices.begin(), op.output_indices.end());
  std::vector<char> Afree, Bfree;
  for (char c : op.output_indices) {
    bool inA = setA.count(c) > 0;
    bool inB = setB.count(c) > 0;
    if (inA && inB) {
      throw std::runtime_error(
        "Unsupported einsum: output label appears in both inputs (batch dims not supported in matmul conversion)");
    }
  }
}

} // namespace

template <typename T>
BlockSparseNumericResult<T> contract_block_sparse_and_accumulate_tblis(const BlockSparseTensor &A, const BlockSparseTensor &B,
                                                                  const EinsumOperation &op,
                                                                  const std::vector<std::vector<T>> &A_blocks_host,
                                                                  const std::vector<std::vector<T>> &B_blocks_host)
{
  
  BlockSparseNumericResult<T> result;

  ensure_disjoint_output_labels(A, B, op);
    
  const LabelGroups g = derive_label_groups(A, B, op);
  BlockSparseTensor D = make_output_tensor(A, B, op);
  // Block lookup maps
  auto aidx = coord_index_map(A, A_blocks_host);
  auto bidx = coord_index_map(B, B_blocks_host);
    
  float kernel_ms = 0.0f;
  std::cout << "D.coordinates (" << D.coordinates.size() << "):\n";
  
  for (const auto &co: D.coordinates) {
    // Build templates that pin free indices to this output coordinate
    auto templA = build_template_for_free_labels(A.indices, g.Arow, D, co);
    auto templB = build_template_for_free_labels(B.indices, g.Brow, D, co);

    // Enumerate K-tuples that exist in both A and B
    auto k_tuples = enumerate_k_tuples(A, B, templA, templB, g.K, g.K);

    int64_t Ktot = 0;
    for (const auto &kt : k_tuples) Ktot += k_extent_product(A, g.K, kt);
    int64_t k_offset = 0;
    // Get extents for D block and create zero-initialized view
    std::vector<int64_t> d_extents;
    int64_t totalD = 1;
    for (size_t i = 0; i < D.indices.size(); i++) {
      char lbl = D.indices[i];
      int sec = co[i];
      d_extents.push_back(D.block_sizes.at(lbl)[sec]);
      totalD *= D.block_sizes.at(lbl)[sec];
    }
    // std::cout << "Total D: " << totalD << std::endl;

    std::vector<T> dataD(totalD, 0);
    MArray::varray_view<T> D_block_view(d_extents, dataD.data());
    tblis::label_vector idx_D;
    for(auto c : op.output_indices) {
      idx_D.push_back(c);
    }

    for (const auto &kt : k_tuples) {
      // Materialize full coordinates for this k-tuple
      std::vector<int> ca = templA;
      for (size_t idx = 0; idx < g.K.size(); ++idx) {
        int ia = -1;
        for (int j = 0; j < (int)A.indices.size(); ++j)
          if (A.indices[j] == g.K[idx]) {
            ia = j;
            break;
        }
        ca[ia] = kt[idx];
      }
      std::vector<int> cb = templB;
      for (size_t idx = 0; idx < g.K.size(); ++idx) {
        int ib = -1;
        for (int j = 0; j < (int)B.indices.size(); ++j)
          if (B.indices[j] == g.K[idx]) {
            ib = j;
            break;
          }
        cb[ib] = kt[idx];
      }
      
      auto it_a = aidx.find(ca);
      auto it_b = bidx.find(cb);

      if (it_a != aidx.end() && it_b != bidx.end()) {
        // Get extents for A block
        int64_t totalA = 1;
        std::vector<int64_t> a_extents;
        for (size_t i = 0; i < A.indices.size(); i++) {
          char lbl = A.indices[i];
          int sec = ca[i];
          a_extents.push_back(A.block_sizes.at(lbl)[sec]);
          totalA *= A.block_sizes.at(lbl)[sec];
        }
        std::vector<T> dataA;
        dataA = A_blocks_host[it_a->second];
        // Convert dataA from column-major to row-major
        std::vector<T> row_major_dataA(totalA);
        std::vector<int64_t> row_strides_A(a_extents.size());
        row_strides_A[row_strides_A.size() - 1] = 1;
        for (int64_t i = row_strides_A.size() - 2; i >= 0; --i) {
          row_strides_A[i] = row_strides_A[i + 1] * a_extents[i + 1];
        }
        for (int64_t i = 0; i < totalA; i++) {
          // Calculate column-major index from row-major index
          int64_t col_idx = i;
          int64_t row_idx = 0;
          for (size_t dim = 0; dim < a_extents.size(); dim++) {
            int64_t coord = col_idx % a_extents[dim];
            col_idx /= a_extents[dim];
            row_idx += coord * row_strides_A[dim];
          }
          row_major_dataA[row_idx] = dataA[i];
        }
        MArray::varray_view<T> A_block_view(a_extents, row_major_dataA.data());
            
        // Get extents for B block
        int64_t totalB = 1;
        std::vector<int64_t> b_extents;
        for (size_t i = 0; i < B.indices.size(); i++) {
          char lbl = B.indices[i];
          int sec = cb[i];
          b_extents.push_back(B.block_sizes.at(lbl)[sec]);
          totalB *= B.block_sizes.at(lbl)[sec];
        }

        std::vector<T> dataB;
        dataB = B_blocks_host[it_b->second];
        // Convert dataA from column-major to row-major
        std::vector<T> row_major_dataB(totalB);
        std::vector<int64_t> row_strides_B(b_extents.size());
        row_strides_B[row_strides_B.size() - 1] = 1;
        for (int64_t i = row_strides_A.size() - 2; i >= 0; --i) {
          row_strides_B[i] = row_strides_B[i + 1] * b_extents[i + 1];
        }
        for (int64_t i = 0; i < totalB; i++) {
          // Calculate column-major index from row-major index
          int64_t col_idx = i;
          int64_t row_idx = 0;
          for (size_t dim = 0; dim < b_extents.size(); dim++) {
            int64_t coord = col_idx % b_extents[dim];
            col_idx /= b_extents[dim];
            row_idx += coord * row_strides_B[dim];
          }
          row_major_dataB[row_idx] = dataB[i];
        }
        MArray::varray_view<T> B_block_view(b_extents, row_major_dataB.data());
        tblis::label_vector idx_A, idx_B;
        for(auto c : op.input_indices[0]) {
          idx_A.push_back(c);
        }
        for(auto c : op.input_indices[1]) {
          idx_B.push_back(c);
        }
        auto start = std::chrono::high_resolution_clock::now();
        tblis::mult<T>(static_cast<T>(1), A_block_view, idx_A.data(), B_block_view, idx_B.data(), static_cast<T>(1), D_block_view, idx_D.data());
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        kernel_ms += static_cast<float>(elapsed.count() * 1000.0);
      }
    }
    // Store D block data back to column-major vector
    // Convert dataD from row-major to column-major order
    std::vector<T> col_major_dataD(totalD);
    for (int64_t i = 0; i < totalD; i++) {
      // Calculate column-major index from row-major index
      int64_t row_major_idx = i;
      int64_t col_major_idx = 0;
      int64_t stride = 1;
      for (size_t dim = 0; dim < d_extents.size(); dim++) {
        int64_t coord = row_major_idx % d_extents[dim];
        row_major_idx /= d_extents[dim];
        col_major_idx += coord * stride;
        stride *= d_extents[dim];
      }
      col_major_dataD[col_major_idx] = dataD[i];
    }
    std::vector<T> block_data(col_major_dataD.begin(), col_major_dataD.end());
    result.blocks.push_back(std::move(block_data));
  }
  result.tensor = D;
  result.kernel_ms = kernel_ms; // Placeholder
  result.flops = compute_blocksparse_actual_flops(A, B, op);
  return result;
}

template BlockSparseNumericResult<float> 
contract_block_sparse_and_accumulate_tblis<float>(const BlockSparseTensor &A, const BlockSparseTensor &B,
                                                                  const EinsumOperation &op,
                                                                  const std::vector<std::vector<float>> &A_blocks_host,
                                                                  const std::vector<std::vector<float>> &B_blocks_host);

template BlockSparseNumericResult<double> 
contract_block_sparse_and_accumulate_tblis<double>(const BlockSparseTensor &A, const BlockSparseTensor &B,
                                                                  const EinsumOperation &op,
                                                                  const std::vector<std::vector<double>> &A_blocks_host,
                                                                  const std::vector<std::vector<double>> &B_blocks_host);
