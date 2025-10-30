// Standalone tool to compute actual FLOPs for two block-sparse tensors.
// Mirrors tools/bs_flops.py behavior for speed and easy integration.

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

struct BSTensor {
  std::string name;
  std::vector<char> indices;                              // e.g., {'i','j'}
  std::vector<int> num_blocks;                            // counts per dimension
  std::unordered_map<char, std::vector<int>> block_sizes; // label -> sizes per section
  std::vector<std::vector<int>> coordinates;              // non-zero block coordinates

  int pos(char lbl) const {
    for (size_t i = 0; i < indices.size(); ++i)
      if (indices[i] == lbl) return static_cast<int>(i);
    return -1;
  }
};

static inline std::string trim(const std::string &s) {
  size_t b = 0, e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

static std::vector<int> parse_brace_list(const std::string &line) {
  auto lpos = line.find('{');
  auto rpos = line.rfind('}');
  if (lpos == std::string::npos || rpos == std::string::npos || rpos <= lpos) {
    throw std::runtime_error("Expected brace-delimited list");
  }
  std::string inner = line.substr(lpos + 1, rpos - lpos - 1);
  std::vector<int> out;
  std::stringstream ss(inner);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    tok = trim(tok);
    if (!tok.empty()) out.push_back(std::stoi(tok));
  }
  return out;
}

static BSTensor read_bs_with_indices(const std::string &path, const std::vector<char> *fallback_indices) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Failed to open file: " + path);
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(in, line)) lines.push_back(line);

  // Header like: "# Tensor A (ij)"
  std::string name;
  std::vector<char> idxs;
  for (const auto &ln : lines) {
    auto p = ln.find("# Tensor ");
    if (p != std::string::npos) {
      // name
      size_t name_start = p + 9;
      size_t name_end = ln.find(" (", name_start);
      if (name_end != std::string::npos) name = trim(ln.substr(name_start, name_end - name_start));
      // indices
      size_t l = ln.find('(', name_end);
      size_t r = (l == std::string::npos) ? std::string::npos : ln.find(')', l + 1);
      if (l != std::string::npos && r != std::string::npos && r > l + 1) {
        auto inner = ln.substr(l + 1, r - l - 1);
        for (char c : inner) {
          if (c != ' ' && c != ',') idxs.push_back(c);
        }
      }
      break;
    }
  }

  std::vector<std::string> brace_lines;
  std::vector<std::vector<int>> coords;
  for (const auto &ln : lines) {
    std::string s = trim(ln);
    if (s.empty() || s.rfind('#', 0) == 0) continue;
    bool has_brace = (s.find('{') != std::string::npos) && (s.find('}') != std::string::npos);
    if (has_brace) {
      brace_lines.push_back(s);
      continue;
    }
    if (s.find(',') != std::string::npos) {
      std::vector<int> c;
      std::stringstream ss(s);
      std::string tok;
      while (std::getline(ss, tok, ',')) {
        tok = trim(tok);
        if (!tok.empty()) c.push_back(std::stoi(tok));
      }
      if (!c.empty()) coords.push_back(std::move(c));
    }
  }

  if (brace_lines.size() < 2) {
    throw std::runtime_error(path + ": Expected at least two brace-delimited lines (sections and sizes)");
  }
  std::vector<int> num_sections = parse_brace_list(brace_lines[0]);
  std::vector<int> sizes_concat = parse_brace_list(brace_lines[1]);

  if (idxs.empty()) {
    if (!fallback_indices || fallback_indices->size() != num_sections.size()) {
      throw std::runtime_error(path + ": Missing header with indices and no valid fallback provided");
    }
    idxs = *fallback_indices;
  }

  std::unordered_map<char, std::vector<int>> block_sizes;
  size_t si = 0;
  for (size_t i = 0; i < idxs.size(); ++i) {
    int cnt = num_sections[i];
    std::vector<int> seg;
    seg.reserve(cnt);
    for (int j = 0; j < cnt; ++j) {
      if (si >= sizes_concat.size()) throw std::runtime_error(path + ": Sizes list too short for declared sections");
      seg.push_back(sizes_concat[si++]);
    }
    block_sizes[idxs[i]] = std::move(seg);
  }

  BSTensor t;
  t.name = name.empty() ? path : name;
  t.indices = idxs;
  t.num_blocks = num_sections;
  t.block_sizes = std::move(block_sizes);
  // Optional dedup if input has repeated coordinates
  if (!coords.empty()) {
    std::sort(coords.begin(), coords.end());
    coords.erase(std::unique(coords.begin(), coords.end()), coords.end());
  }
  t.coordinates = std::move(coords);
  return t;
}

struct EinsumOp {
  std::vector<char> Aidx;
  std::vector<char> Bidx;
  std::vector<char> Oidx;
  std::vector<char> Kidx;
};

static EinsumOp parse_einsum(const std::string &e) {
  if (e == "infer") return EinsumOp{}; // handled later
  auto arrow = e.find("->");
  auto comma = e.find(',');
  if (arrow == std::string::npos || comma == std::string::npos || comma > arrow) {
    throw std::invalid_argument("Einsum must be in form 'Aidx,Bidx->Oidx'");
  }
  std::string a = e.substr(0, comma);
  std::string b = e.substr(comma + 1, arrow - (comma + 1));
  std::string o = e.substr(arrow + 2);
  EinsumOp op;
  for (char c : a)
    if (c != ' ') op.Aidx.push_back(c);
  for (char c : b)
    if (c != ' ') op.Bidx.push_back(c);
  for (char c : o)
    if (c != ' ') op.Oidx.push_back(c);
  std::set<char> out(op.Oidx.begin(), op.Oidx.end());
  std::unordered_map<char, int> cnt;
  for (char c : op.Aidx) cnt[c]++;
  for (char c : op.Bidx) cnt[c]++;
  for (auto &kv : cnt)
    if (!out.count(kv.first) && kv.second > 0) op.Kidx.push_back(kv.first);
  return op;
}

static std::vector<int> read_index_mapping(const std::string &path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Failed to open file: " + path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.find("# Index mapping") != std::string::npos) {
      std::string next;
      if (!std::getline(in, next)) break;
      // Strip leading comment if present
      auto pos = next.find('#');
      if (pos != std::string::npos) next = next.substr(pos + 1);
      std::vector<int> mapping;
      std::stringstream ss(next);
      std::string tok;
      while (std::getline(ss, tok, ',')) {
        tok = trim(tok);
        if (tok.empty()) continue;
        auto colon = tok.find(':');
        if (colon == std::string::npos) continue;
        std::string val = trim(tok.substr(colon + 1));
        if (!val.empty()) mapping.push_back(std::stoi(val));
      }
      return mapping;
    }
  }
  throw std::runtime_error("Index mapping not found in " + path);
}

static char next_label_from_counter(int k) { return static_cast<char>('a' + (k % 26)); }

static EinsumOp infer_einsum_from_mapping(const std::string &a_path, const std::string &b_path) {
  auto ma = read_index_mapping(a_path);
  auto mb = read_index_mapping(b_path);

  std::unordered_map<int, char> pos_label;
  std::unordered_map<int, char> neg_label;
  auto get_pos = [&](int id, int &k) -> char {
    auto it = pos_label.find(id);
    if (it != pos_label.end()) return it->second;
    char c = next_label_from_counter(k++);
    pos_label[id] = c;
    return c;
  };
  auto get_neg = [&](int id, int &k) -> char {
    auto it = neg_label.find(id);
    if (it != neg_label.end()) return it->second;
    char c = next_label_from_counter(k++);
    neg_label[id] = c;
    return c;
  };

  EinsumOp op;
  int k = 0;
  for (int id : ma) op.Aidx.push_back(id < 0 ? get_neg(-id, k) : get_pos(id, k));
  for (int id : mb) op.Bidx.push_back(id < 0 ? get_neg(-id, k) : get_pos(id, k));

  std::set<char> seen;
  for (int id : ma)
    if (id > 0) {
      char c = pos_label[id];
      if (!seen.count(c)) {
        op.Oidx.push_back(c);
        seen.insert(c);
      }
    }
  for (int id : mb)
    if (id > 0) {
      char c = pos_label[id];
      if (!seen.count(c)) {
        op.Oidx.push_back(c);
        seen.insert(c);
      }
    }

  std::set<char> out(op.Oidx.begin(), op.Oidx.end());
  std::unordered_map<char, int> cnt;
  for (char c : op.Aidx) cnt[c]++;
  for (char c : op.Bidx) cnt[c]++;
  for (auto &kv : cnt)
    if (!out.count(kv.first) && kv.second > 0) op.Kidx.push_back(kv.first);
  return op;
}

static void ensure_disjoint_output_labels(const BSTensor &A, const BSTensor &B, const EinsumOp &op) {
  std::set<char> setA(A.indices.begin(), A.indices.end());
  std::set<char> setB(B.indices.begin(), B.indices.end());
  for (char c : op.Oidx) {
    if (setA.count(c) && setB.count(c)) {
      throw std::runtime_error("Unsupported einsum: output label appears in both inputs (batch dims not supported)");
    }
  }
}

static void remap_indices_inplace(BSTensor &T, const std::vector<char> &new_indices) {
  if (new_indices.size() != T.indices.size()) return; // ignore if mismatch
  // Rebuild block_sizes with new keys by positional order
  std::unordered_map<char, std::vector<int>> new_map;
  for (size_t i = 0; i < new_indices.size(); ++i) {
    char old_lbl = T.indices[i];
    char new_lbl = new_indices[i];
    auto it = T.block_sizes.find(old_lbl);
    if (it == T.block_sizes.end()) throw std::runtime_error("Missing block_sizes for old label during remap");
    new_map.emplace(new_lbl, it->second);
  }
  T.indices = new_indices;
  T.block_sizes = std::move(new_map);
}

static long long compute_actual_flops(const BSTensor &A, const BSTensor &B, const EinsumOp &op) {
  ensure_disjoint_output_labels(A, B, op);

  // Partition output labels
  std::vector<char> Arow, Brow;
  for (char c : op.Oidx) {
    (std::find(A.indices.begin(), A.indices.end(), c) != A.indices.end()) ? Arow.push_back(c) : Brow.push_back(c);
  }

  // Fast index lookup tables
  int posA[256];
  int posB[256];
  std::fill(std::begin(posA), std::end(posA), -1);
  std::fill(std::begin(posB), std::end(posB), -1);
  for (size_t i = 0; i < A.indices.size(); ++i) posA[static_cast<unsigned char>(A.indices[i])] = static_cast<int>(i);
  for (size_t i = 0; i < B.indices.size(); ++i) posB[static_cast<unsigned char>(B.indices[i])] = static_cast<int>(i);

  // Prebind block size vectors for quick access
  std::vector<const std::vector<int> *> Arow_sizes;
  Arow_sizes.reserve(Arow.size());
  for (char lbl : Arow) Arow_sizes.push_back(&A.block_sizes.at(lbl));
  std::vector<const std::vector<int> *> Brow_sizes;
  Brow_sizes.reserve(Brow.size());
  for (char lbl : Brow) Brow_sizes.push_back(&B.block_sizes.at(lbl));
  std::vector<const std::vector<int> *> K_sizesA;
  K_sizesA.reserve(op.Kidx.size());
  for (char k : op.Kidx) K_sizesA.push_back(&A.block_sizes.at(k));

  auto make_kt_key = [&](const std::vector<int> &coord, const int *posTbl) -> std::string {
    std::string key;
    key.reserve(4 * op.Kidx.size());
    for (char k : op.Kidx) {
      int sec = coord[posTbl[static_cast<unsigned char>(k)]];
      uint32_t v = static_cast<uint32_t>(sec);
      key.push_back(static_cast<char>(v & 0xFF));
      key.push_back(static_cast<char>((v >> 8) & 0xFF));
      key.push_back(static_cast<char>((v >> 16) & 0xFF));
      key.push_back(static_cast<char>((v >> 24) & 0xFF));
    }
    return key;
  };

  // Group by K-tuple: accumulate sum of M (for A) and sum of N (for B)
  std::unordered_map<std::string, long long> sumA;
  sumA.reserve(A.coordinates.size());
  std::unordered_map<std::string, long long> sumB;
  sumB.reserve(B.coordinates.size());
  std::unordered_map<std::string, long long> Ksize;
  Ksize.reserve((A.coordinates.size() + B.coordinates.size()) / 2);

  // Build A sums and K sizes
  for (const auto &ca : A.coordinates) {
    long long MA = 1;
    for (size_t i = 0; i < Arow.size(); ++i) {
      char lbl = Arow[i];
      int sec = ca[posA[static_cast<unsigned char>(lbl)]];
      MA *= (*Arow_sizes[i])[sec];
    }
    std::string key = make_kt_key(ca, posA);
    auto it = sumA.find(key);
    if (it == sumA.end()) {
      sumA.emplace(key, MA);
      // Compute Ksize for this kt from A sizes once
      long long ks = 1;
      size_t idx = 0;
      for (char k : op.Kidx) {
        int sec = ca[posA[static_cast<unsigned char>(k)]];
        ks *= (*K_sizesA[idx++])[sec];
      }
      Ksize.emplace(key, ks);
    } else {
      it->second += MA;
    }
  }

  // Build B sums
  for (const auto &cb : B.coordinates) {
    long long NB = 1;
    for (size_t i = 0; i < Brow.size(); ++i) {
      char lbl = Brow[i];
      int sec = cb[posB[static_cast<unsigned char>(lbl)]];
      NB *= (*Brow_sizes[i])[sec];
    }
    std::string key = make_kt_key(cb, posB);
    auto it = sumB.find(key);
    if (it == sumB.end()) {
      sumB.emplace(key, NB);
    } else {
      it->second += NB;
    }
  }

  // Sum over intersection of K-tuples
  long long total = 0;
  for (const auto &kv : sumA) {
    auto itB = sumB.find(kv.first);
    if (itB == sumB.end()) continue;
    long long ks = Ksize[kv.first];
    total += 2LL * ks * kv.second * itB->second;
  }
  return total;
}

static long long compute_unique_bytes(const BSTensor &A, const BSTensor &B, const EinsumOp &op, int elem_bytes) {
  ensure_disjoint_output_labels(A, B, op);

  // Label position lookup
  int posA[256];
  int posB[256];
  std::fill(std::begin(posA), std::end(posA), -1);
  std::fill(std::begin(posB), std::end(posB), -1);
  for (size_t i = 0; i < A.indices.size(); ++i) posA[static_cast<unsigned char>(A.indices[i])] = static_cast<int>(i);
  for (size_t i = 0; i < B.indices.size(); ++i) posB[static_cast<unsigned char>(B.indices[i])] = static_cast<int>(i);

  // Partition output labels
  std::vector<char> Arow, Brow;
  for (char c : op.Oidx) { (posA[static_cast<unsigned char>(c)] >= 0) ? Arow.push_back(c) : Brow.push_back(c); }

  // Map label to index in Arow/Brow for quick access during output key building
  int idxArow[256];
  int idxBrow[256];
  std::fill(std::begin(idxArow), std::end(idxArow), -1);
  std::fill(std::begin(idxBrow), std::end(idxBrow), -1);
  for (size_t i = 0; i < Arow.size(); ++i) idxArow[static_cast<unsigned char>(Arow[i])] = static_cast<int>(i);
  for (size_t i = 0; i < Brow.size(); ++i) idxBrow[static_cast<unsigned char>(Brow[i])] = static_cast<int>(i);

  // A bytes: sum of unique A block payloads once
  long long a_bytes = 0;
  for (const auto &ca : A.coordinates) {
    long long elems = 1;
    for (char lbl : A.indices) {
      int sec = ca[posA[static_cast<unsigned char>(lbl)]];
      elems *= A.block_sizes.at(lbl)[sec];
    }
    a_bytes += elems * elem_bytes;
  }

  // B bytes: sum of unique B block payloads once
  long long b_bytes = 0;
  for (const auto &cb : B.coordinates) {
    long long elems = 1;
    for (char lbl : B.indices) {
      int sec = cb[posB[static_cast<unsigned char>(lbl)]];
      elems *= B.block_sizes.at(lbl)[sec];
    }
    b_bytes += elems * elem_bytes;
  }

  // Group A and B by K-tuple
  auto make_kt_key = [&](const std::vector<int> &coord, const int *posTbl) -> std::string {
    std::string key;
    key.reserve(4 * op.Kidx.size());
    for (char k : op.Kidx) {
      int sec = coord[posTbl[static_cast<unsigned char>(k)]];
      uint32_t v = static_cast<uint32_t>(sec);
      key.push_back(static_cast<char>(v & 0xFF));
      key.push_back(static_cast<char>((v >> 8) & 0xFF));
      key.push_back(static_cast<char>((v >> 16) & 0xFF));
      key.push_back(static_cast<char>((v >> 24) & 0xFF));
    }
    return key;
  };

  struct RowRec {
    std::vector<int> sec;
    long long prod;
  };
  std::unordered_map<std::string, std::vector<RowRec>> groupsA;
  groupsA.reserve(A.coordinates.size());
  std::unordered_map<std::string, std::vector<RowRec>> groupsB;
  groupsB.reserve(B.coordinates.size());

  // Build A groups with Arow sections and M products
  for (const auto &ca : A.coordinates) {
    std::string key = make_kt_key(ca, posA);
    RowRec r;
    r.sec.reserve(Arow.size());
    r.prod = 1;
    for (char lbl : Arow) {
      int sec = ca[posA[static_cast<unsigned char>(lbl)]];
      r.sec.push_back(sec);
      r.prod *= A.block_sizes.at(lbl)[sec];
    }
    groupsA[key].push_back(std::move(r));
  }
  // Build B groups with Brow sections and N products
  for (const auto &cb : B.coordinates) {
    std::string key = make_kt_key(cb, posB);
    RowRec r;
    r.sec.reserve(Brow.size());
    r.prod = 1;
    for (char lbl : Brow) {
      int sec = cb[posB[static_cast<unsigned char>(lbl)]];
      r.sec.push_back(sec);
      r.prod *= B.block_sizes.at(lbl)[sec];
    }
    groupsB[key].push_back(std::move(r));
  }

  // Unique output coordinates across all K-tuples
  std::unordered_set<std::string> out_keys;
  out_keys.reserve(A.coordinates.size() + B.coordinates.size());

  long long c_bytes = 0;
  const size_t out_dims = op.Oidx.size();
  auto encode_out_key = [&](const RowRec &ra, const RowRec &rb) -> std::string {
    std::string key;
    key.resize(out_dims * 4);
    size_t offset = 0;
    for (char lbl : op.Oidx) {
      int sec = -1;
      int ia = idxArow[static_cast<unsigned char>(lbl)];
      if (ia >= 0)
        sec = ra.sec[ia];
      else {
        int ib = idxBrow[static_cast<unsigned char>(lbl)];
        sec = rb.sec[ib];
      }
      uint32_t v = static_cast<uint32_t>(sec);
      key[offset + 0] = static_cast<char>(v & 0xFF);
      key[offset + 1] = static_cast<char>((v >> 8) & 0xFF);
      key[offset + 2] = static_cast<char>((v >> 16) & 0xFF);
      key[offset + 3] = static_cast<char>((v >> 24) & 0xFF);
      offset += 4;
    }
    return key;
  };

  // For each K group intersection, enumerate unique outputs and tally bytes once per output block
  for (const auto &kv : groupsA) {
    auto itB = groupsB.find(kv.first);
    if (itB == groupsB.end()) continue;
    const auto &avec = kv.second;
    const auto &bvec = itB->second;
    for (const auto &ra : avec) {
      for (const auto &rb : bvec) {
        std::string ok = encode_out_key(ra, rb);
        auto ins = out_keys.insert(ok);
        if (ins.second) { c_bytes += ra.prod * rb.prod * elem_bytes; }
      }
    }
  }

  return a_bytes + b_bytes + c_bytes;
}

int main(int argc, char **argv) {
  // Optional flags
  int elem_bytes = 4; // default f32
  enum class Report { FLOPS, BYTES, BOTH };
  Report report = Report::FLOPS;

  int idx = 1;
  auto starts_with = [](const char *s, const char *p) { return std::strncmp(s, p, std::strlen(p)) == 0; };
  while (idx < argc && argv[idx][0] == '-') {
    if (starts_with(argv[idx], "--elem-bytes=")) {
      elem_bytes = std::atoi(argv[idx] + std::strlen("--elem-bytes="));
      ++idx;
      continue;
    }
    if (starts_with(argv[idx], "--dtype=")) {
      std::string dt = argv[idx] + std::strlen("--dtype=");
      if (dt == "f32")
        elem_bytes = 4;
      else if (dt == "f64")
        elem_bytes = 8;
      else {
        std::cerr << "Unknown dtype: " << dt << std::endl;
        return 1;
      }
      ++idx;
      continue;
    }
    if (starts_with(argv[idx], "--report=")) {
      std::string rp = argv[idx] + std::strlen("--report=");
      if (rp == "flops")
        report = Report::FLOPS;
      else if (rp == "bytes")
        report = Report::BYTES;
      else if (rp == "both")
        report = Report::BOTH;
      else {
        std::cerr << "Unknown report: " << rp << std::endl;
        return 1;
      }
      ++idx;
      continue;
    }
    break;
  }
  if (argc - idx != 3) {
    std::cerr
      << "Usage: bs_flops [--elem-bytes=N|--dtype=f32|f64] [--report=flops|bytes|both] '<einsum|infer>' <A.bs> <B.bs>"
      << std::endl;
    return 1;
  }
  std::string einsum = argv[idx + 0];
  std::string Apath = argv[idx + 1];
  std::string Bpath = argv[idx + 2];

  try {
    EinsumOp op = parse_einsum(einsum);
    if (einsum == "infer") { op = infer_einsum_from_mapping(Apath, Bpath); }
    // Read tensors; allow fallback indices if headers missing
    BSTensor A = read_bs_with_indices(Apath, op.Aidx.empty() ? nullptr : &op.Aidx);
    BSTensor B = read_bs_with_indices(Bpath, op.Bidx.empty() ? nullptr : &op.Bidx);
    if (!op.Aidx.empty() && op.Aidx.size() != A.indices.size()) {
      throw std::runtime_error("Rank mismatch: einsum A has " + std::to_string(op.Aidx.size()) +
                               " dims but tensor A file has " + std::to_string(A.indices.size()));
    }
    if (!op.Bidx.empty() && op.Bidx.size() != B.indices.size()) {
      throw std::runtime_error("Rank mismatch: einsum B has " + std::to_string(op.Bidx.size()) +
                               " dims but tensor B file has " + std::to_string(B.indices.size()));
    }
    // If headers were present but user provided different label names with same rank,
    // remap tensor indices to match einsum labels positionally to avoid label mismatches.
    if (!op.Aidx.empty() && op.Aidx.size() == A.indices.size()) remap_indices_inplace(A, op.Aidx);
    if (!op.Bidx.empty() && op.Bidx.size() == B.indices.size()) remap_indices_inplace(B, op.Bidx);
    long long flops = compute_actual_flops(A, B, op);
    if (report == Report::FLOPS) {
      std::cout << flops << std::endl;
    } else if (report == Report::BYTES) {
      long long bytes = compute_unique_bytes(A, B, op, elem_bytes);
      std::cout << bytes << std::endl;
    } else {
      long long bytes = compute_unique_bytes(A, B, op, elem_bytes);
      std::cout << flops << "\n" << bytes << std::endl;
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 2;
  }
  return 0;
}
