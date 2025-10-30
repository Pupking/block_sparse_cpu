#include "tensor_layout.h"

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>

std::set<std::vector<int>>
build_output_coordinates(const BlockSparseTensor &A, const BlockSparseTensor &B, const EinsumOperation &op) {
  std::map<char, int> posA;
  std::map<char, int> posB;
  for (int i = 0; i < (int)A.indices.size(); ++i) posA[A.indices[i]] = i;
  for (int i = 0; i < (int)B.indices.size(); ++i) posB[B.indices[i]] = i;

  std::set<std::vector<int>> out_coords;
  for (const auto &ca : A.coordinates) {
    for (const auto &cb : B.coordinates) {
      bool compatible = true;
      for (char k : op.contraction_indices) {
        int ia = posA.at(k);
        int ib = posB.at(k);
        if (ca[ia] != cb[ib]) {
          compatible = false;
          break;
        }
      }
      if (!compatible) continue;

      std::vector<int> co(op.output_indices.size());
      for (size_t i = 0; i < op.output_indices.size(); ++i) {
        char lbl = op.output_indices[i];
        auto ita = posA.find(lbl);
        if (ita != posA.end()) {
          co[i] = ca[ita->second];
          continue;
        }
        auto itb = posB.find(lbl);
        if (itb != posB.end()) co[i] = cb[itb->second];
      }
      out_coords.insert(std::move(co));
    }
  }
  return out_coords;
}

namespace {

void ensure_disjoint_output_labels(const BlockSparseTensor &A, const BlockSparseTensor &B, const EinsumOperation &op) {
  std::set<char> setA(A.indices.begin(), A.indices.end());
  std::set<char> setB(B.indices.begin(), B.indices.end());
  for (char c : op.output_indices) {
    bool inA = setA.count(c) > 0;
    bool inB = setB.count(c) > 0;
    if (inA && inB) {
      throw std::runtime_error("Unsupported einsum: output index appears in both inputs");
    }
  }
}

} // namespace

long double compute_blocksparse_actual_flops(const BlockSparseTensor &A, const BlockSparseTensor &B,
                                             const EinsumOperation &op) {
  ensure_disjoint_output_labels(A, B, op);

  std::vector<char> Arow;
  std::vector<char> Brow;
  for (char c : op.output_indices) {
    bool inA = std::find(A.indices.begin(), A.indices.end(), c) != A.indices.end();
    if (inA)
      Arow.push_back(c);
    else
      Brow.push_back(c);
  }
  std::vector<char> Ak = op.contraction_indices;
  std::vector<char> Bk = Ak;

  BlockSparseTensor D;
  D.name = "Output";
  D.indices = op.output_indices;
  for (char lbl : D.indices) {
    const auto &src = (std::find(A.indices.begin(), A.indices.end(), lbl) != A.indices.end()) ? A : B;
    D.num_blocks.push_back((int)src.block_sizes.at(lbl).size());
    D.block_sizes[lbl] = src.block_sizes.at(lbl);
  }
  D.coordinates = build_output_coordinates(A, B, op);

  auto k_extents = [&](const std::vector<int> &kt, const BlockSparseTensor &Ten, const std::vector<char> &klabels) {
    int64_t p = 1;
    for (size_t idx = 0; idx < klabels.size(); ++idx) {
      char lbl = klabels[idx];
      int sec = kt[idx];
      p *= Ten.block_sizes.at(lbl)[sec];
    }
    return p;
  };

  long double total_flops = 0.0L;
  for (const auto &co : D.coordinates) {
    int64_t M = 1;
    for (char lbl : Arow) {
      int posOut = -1;
      for (int i = 0; i < (int)D.indices.size(); ++i)
        if (D.indices[i] == lbl) {
          posOut = i;
          break;
        }
      int sec = co[posOut];
      M *= D.block_sizes.at(lbl)[sec];
    }

    int64_t N = 1;
    for (char lbl : Brow) {
      int posOut = -1;
      for (int i = 0; i < (int)D.indices.size(); ++i)
        if (D.indices[i] == lbl) {
          posOut = i;
          break;
        }
      int sec = co[posOut];
      N *= D.block_sizes.at(lbl)[sec];
    }

    std::vector<int> templA(A.indices.size(), -1);
    std::vector<int> templB(B.indices.size(), -1);
    for (char lbl : Arow) {
      int ia = -1;
      for (int i = 0; i < (int)A.indices.size(); ++i)
        if (A.indices[i] == lbl) {
          ia = i;
          break;
        }
      int io = -1;
      for (int i = 0; i < (int)D.indices.size(); ++i)
        if (D.indices[i] == lbl) {
          io = i;
          break;
        }
      templA[ia] = co[io];
    }
    for (char lbl : Brow) {
      int ib = -1;
      for (int i = 0; i < (int)B.indices.size(); ++i)
        if (B.indices[i] == lbl) {
          ib = i;
          break;
        }
      int io = -1;
      for (int i = 0; i < (int)D.indices.size(); ++i)
        if (D.indices[i] == lbl) {
          io = i;
          break;
        }
      templB[ib] = co[io];
    }

    std::set<std::vector<int>> k_tuples;
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

    int64_t Ktot = 0;
    for (const auto &kt : k_tuples) {
      int64_t ka = k_extents(kt, A, Ak);
      int64_t kb = k_extents(kt, B, Bk);
      if (ka != kb) {
        throw std::runtime_error("Mismatched K segment extents between A and B");
      }
      Ktot += ka;
    }

    total_flops += 2.0L * static_cast<long double>(M) * static_cast<long double>(N) * static_cast<long double>(Ktot);
  }

  return total_flops;
}
