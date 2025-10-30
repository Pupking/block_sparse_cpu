#pragma once

#include "block_sparse_tensor.h"
#include "validation.h" // for DenseHostTensor

#include <cmath>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

#ifndef MAX_TENSOR_RANK
#define MAX_TENSOR_RANK 8
#endif

namespace tensor_utils {

// Map labels to extents across A and B, ensuring consistency where labels collide.
inline std::map<char, int64_t> label_extents_map(const std::vector<char> &Aidx, const std::vector<char> &Bidx,
                                                 const std::vector<int64_t> &Aext, const std::vector<int64_t> &Bext) {
  std::map<char, int64_t> m;
  for (size_t i = 0; i < Aidx.size(); ++i) m[Aidx[i]] = Aext[i];
  for (size_t i = 0; i < Bidx.size(); ++i) {
    auto it = m.find(Bidx[i]);
    if (it == m.end())
      m[Bidx[i]] = Bext[i];
    else if (it->second != Bext[i])
      throw std::runtime_error("Mismatched extents for label between A and B");
  }
  return m;
}

inline long double product_ld(const std::vector<int64_t> &v) {
  long double p = 1.0L;
  for (auto x : v) p *= static_cast<long double>(x);
  return p;
}

// Compute dense-equivalent FLOPs for a binary einsum: 2 * prod(output extents) * prod(contraction extents)
inline long double dense_flop_count(const EinsumOperation &op, const std::vector<char> &Aidx,
                                    const std::vector<char> &Bidx, const std::vector<int64_t> &Aext,
                                    const std::vector<int64_t> &Bext) {
  auto lm = label_extents_map(Aidx, Bidx, Aext, Bext);
  std::vector<int64_t> out_e;
  out_e.reserve(op.output_indices.size());
  std::vector<int64_t> con_e;
  con_e.reserve(op.contraction_indices.size());
  for (char c : op.output_indices) out_e.push_back(lm.at(c));
  for (char c : op.contraction_indices) con_e.push_back(lm.at(c));
  return 2.0L * product_ld(out_e) * product_ld(con_e);
}

template <typename T>
inline void print_error_metrics(const DenseHostTensor<T> &ref, const DenseHostTensor<T> &got, double eps,
                                const char *tag) {
  if (ref.extents != got.extents) {
    std::cerr << "Extent mismatch in verification (" << tag << ")" << std::endl;
    return;
  }
  long double max_abs = 0.0L, max_rel = 0.0L, l2 = 0.0L;
  size_t elems = ref.data.size();
  for (size_t i = 0; i < elems; ++i) {
    long double a = static_cast<long double>(ref.data[i]);
    long double b = static_cast<long double>(got.data[i]);
    long double d = fabsl(a - b);
    max_abs = std::max(max_abs, d);
    long double denom = std::max(static_cast<long double>(eps), fabsl(a));
    max_rel = std::max(max_rel, d / denom);
    l2 += d * d;
  }
  l2 = std::sqrt(l2);
  std::cout << tag << ": max_abs=" << static_cast<double>(max_abs) << ", max_rel=" << static_cast<double>(max_rel)
            << ", l2=" << static_cast<double>(l2) << std::endl;
}

inline double gflops_from(long double flop, long double ms) {
  return (ms > 0.0L) ? static_cast<double>((flop / 1.0e9L) / (ms / 1.0e3L)) : 0.0;
}

} // namespace tensor_utils
