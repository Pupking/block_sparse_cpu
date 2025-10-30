#include "block_sparse_tensor.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

// BlockSparseTensor (CPU-only)
BlockSparseTensor::BlockSparseTensor() = default;

BlockSparseTensor::BlockSparseTensor(const std::string &n, const std::vector<char> &idx) : name(n), indices(idx) {}

void BlockSparseTensor::initialize(const std::string &tensor_name, const std::vector<char> &tensor_indices,
                                   const std::vector<int> &dimension_blocks,
                                   const std::map<char, std::vector<int>> &dimension_block_sizes,
                                   const std::set<std::vector<int>> &coords) {
  name = tensor_name;
  indices = tensor_indices;
  num_blocks = dimension_blocks;
  block_sizes = dimension_block_sizes;
  coordinates = coords;
}

std::vector<int> BlockSparseTensor::get_total_dimensions() const {
  std::vector<int> dims(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    char dim_idx = indices[i];
    dims[i] = std::accumulate(block_sizes.at(dim_idx).begin(), block_sizes.at(dim_idx).end(), 0);
  }
  return dims;
}

std::vector<int> BlockSparseTensor::get_block_size(const std::vector<int> &coord) const {
  std::vector<int> sizes(coord.size());
  for (size_t i = 0; i < coord.size(); ++i) {
    char dim_idx = indices[i];
    sizes[i] = block_sizes.at(dim_idx)[coord[i]];
  }
  return sizes;
}

bool BlockSparseTensor::has_coordinate(const std::vector<int> &coord) const {
  return coordinates.find(coord) != coordinates.end();
}

int BlockSparseTensor::get_dimension_index(char label) const {
  auto it = std::find(indices.begin(), indices.end(), label);
  if (it != indices.end()) return static_cast<int>(std::distance(indices.begin(), it));
  return -1;
}

void BlockSparseTensor::print_info() const {
  std::cout << "Tensor " << name << " (";
  for (size_t i = 0; i < indices.size(); ++i) {
    std::cout << indices[i];
    if (i < indices.size() - 1) std::cout << "";
  }
  std::cout << ")" << std::endl;

  std::cout << "  Dimensions: ";
  for (size_t i = 0; i < num_blocks.size(); ++i) {
    std::cout << num_blocks[i];
    if (i < num_blocks.size() - 1) std::cout << " x ";
  }
  std::cout << " blocks" << std::endl;
  std::cout << "  Block sizes by dimension:" << std::endl;
  for (size_t i = 0; i < indices.size(); ++i) {
    char dim_idx = indices[i];
    std::cout << "    " << dim_idx << ": ";
    const auto& sizes = block_sizes.at(dim_idx);
    for (size_t j = 0; j < sizes.size(); ++j) {
      std::cout << sizes[j];
      if (j < sizes.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;
  }

  std::cout << "  Coordinates:" << std::endl;
  for (const auto& coord : coordinates) {
    std::cout << "    (";
    for (size_t i = 0; i < coord.size(); ++i) {
      std::cout << coord[i];
      if (i < coord.size() - 1) std::cout << ", ";
    }
    std::cout << ")" << std::endl;
  }

  std::cout << "  Non-zero blocks: " << coordinates.size() << std::endl;
}

// EinsumOperation implementation
void EinsumOperation::parse(const std::string &einsum_str) {
  operation_string = einsum_str;
  input_indices.clear();
  output_indices.clear();
  contraction_indices.clear();

  // Split on "->"
  size_t arrow_pos = einsum_str.find("->");
  if (arrow_pos == std::string::npos) { throw std::invalid_argument("Invalid einsum string: missing '->'"); }

  std::string inputs_str = einsum_str.substr(0, arrow_pos);
  std::string output_str = einsum_str.substr(arrow_pos + 2);

  // Parse input tensors
  std::stringstream ss(inputs_str);
  std::string tensor_str;
  while (std::getline(ss, tensor_str, ',')) {
    std::vector<char> tensor_indices;
    for (char c : tensor_str) {
      if (c != ' ') { tensor_indices.push_back(c); }
    }
    input_indices.push_back(tensor_indices);
  }

  // Parse output tensor
  for (char c : output_str) {
    if (c != ' ') { output_indices.push_back(c); }
  }

  // Determine contraction indices (present in inputs but not in output)
  std::set<char> output_set(output_indices.begin(), output_indices.end());
  std::map<char, int> counts;
  for (const auto &t : input_indices)
    for (char c : t) counts[c]++;
  contraction_indices.clear();
  for (const auto &kv : counts) {
    if (!output_set.count(kv.first) && kv.second > 0) contraction_indices.push_back(kv.first);
  }
}

std::set<char> EinsumOperation::get_unique_indices() const {
  std::set<char> unique;
  for (const auto &tensor_indices : input_indices) {
    for (char idx : tensor_indices) { unique.insert(idx); }
  }
  for (char idx : output_indices) { unique.insert(idx); }
  return unique;
}

bool EinsumOperation::is_valid() const {
  if (input_indices.size() != 2) return false;
  const auto &A = input_indices[0];
  const auto &B = input_indices[1];
  if (A.empty() || B.empty() || output_indices.empty()) return false;

  auto has_unique = [](const std::vector<char> &idx) {
    std::set<char> s(idx.begin(), idx.end());
    return s.size() == idx.size();
  };
  if (!has_unique(A) || !has_unique(B) || !has_unique(output_indices)) return false;

  auto contains = [](const std::vector<char> &idx, char c) {
    return std::find(idx.begin(), idx.end(), c) != idx.end();
  };

  for (char c : output_indices) {
    bool inA = contains(A, c);
    bool inB = contains(B, c);
    if ((inA && inB) || (!inA && !inB)) return false;
  }

  std::set<char> expected_contraction;
  for (char c : A)
    if (contains(B, c)) expected_contraction.insert(c);

  std::set<char> actual_contraction(contraction_indices.begin(), contraction_indices.end());
  if (expected_contraction != actual_contraction) return false;

  for (char c : A)
    if (!contains(B, c) && !contains(output_indices, c)) return false;
  for (char c : B)
    if (!contains(A, c) && !contains(output_indices, c)) return false;

  return true;
}

void EinsumOperation::print_info() const {
  std::cout << "Einsum Operation: " << operation_string << std::endl;
  std::cout << "  Input tensors: " << input_indices.size() << std::endl;
  for (size_t i = 0; i < input_indices.size(); ++i) {
    std::cout << "    Tensor " << i << ": ";
    for (char idx : input_indices[i]) { std::cout << idx; }
    std::cout << std::endl;
  }
  std::cout << "  Output tensor: ";
  for (char idx : output_indices) { std::cout << idx; }
  std::cout << std::endl;
  std::cout << "  Contraction indices: ";
  for (char idx : contraction_indices) { std::cout << idx << " "; }
  std::cout << std::endl;
}

// Parse tensor file function
BlockSparseTensor parse_tensor_file(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) { throw std::runtime_error("Could not open file: " + filename); }

  std::string line;
  std::vector<std::string> lines;

  // Read all lines
  while (std::getline(file, line)) { lines.push_back(line); }
  file.close();

  // Parse tensor metadata
  std::string tensor_name;
  std::vector<char> tensor_indices;
  std::vector<int> dimension_blocks;
  std::map<char, std::vector<int>> dimension_block_sizes;
  std::set<std::vector<int>> coords;

  for (size_t i = 0; i < lines.size(); ++i) {
    const std::string &current_line = lines[i];

    if (current_line.empty()) { continue; }

    // Parse tensor name from comment
    if (current_line.find("# Tensor ") != std::string::npos) {
      size_t name_start = current_line.find("# Tensor ") + 9;
      size_t name_end = current_line.find(" (");
      if (name_end != std::string::npos) { tensor_name = current_line.substr(name_start, name_end - name_start); }

      // Parse indices from parentheses
      size_t indices_start = current_line.find("(") + 1;
      size_t indices_end = current_line.find(")");
      if (indices_end != std::string::npos) {
        std::string indices_str = current_line.substr(indices_start, indices_end - indices_start);
        for (char c : indices_str) {
          if (c != ' ' && c != ',') { tensor_indices.push_back(c); }
        }
      }
    }

    // Skip other comments
    if (!current_line.empty() && current_line[0] == '#') { continue; }

    // Parse dimension blocks (first brace-delimited line)
    if (current_line.find('{') != std::string::npos && current_line.find('}') != std::string::npos &&
        dimension_blocks.empty()) {
      std::string dim_str =
        current_line.substr(current_line.find('{') + 1, current_line.find('}') - current_line.find('{') - 1);

      std::istringstream dim_stream(dim_str);
      std::string token;
      while (std::getline(dim_stream, token, ',')) {
        token.erase(0, token.find_first_not_of(" \t"));
        token.erase(token.find_last_not_of(" \t") + 1);
        if (!token.empty()) dimension_blocks.push_back(std::stoi(token));
      }
    }
    // Skip nnz line (we count coordinates directly)
    else if (current_line.find('{') == std::string::npos && current_line.find(',') == std::string::npos &&
             !dimension_blocks.empty() && dimension_block_sizes.empty()) {
      continue;
    }
    // Parse block sizes (second brace-delimited line)
    else if (current_line.find('{') != std::string::npos && current_line.find('}') != std::string::npos &&
             !dimension_blocks.empty()) {
      std::string sizes_str =
        current_line.substr(current_line.find('{') + 1, current_line.find('}') - current_line.find('{') - 1);

      std::istringstream sizes_stream(sizes_str);
      std::string token;
      std::vector<int> all_sizes;

      while (std::getline(sizes_stream, token, ',')) {
        token.erase(0, token.find_first_not_of(" \t"));
        token.erase(token.find_last_not_of(" \t") + 1);
        if (!token.empty()) all_sizes.push_back(std::stoi(token));
      }

      // Distribute sizes to dimensions
      size_t size_idx = 0;
      for (size_t dim = 0; dim < tensor_indices.size(); ++dim) {
        char idx = tensor_indices[dim];
        int num_blocks_in_dim = dimension_blocks[dim];

        std::vector<int> dim_sizes;
        for (int j = 0; j < num_blocks_in_dim; ++j) {
          if (size_idx < all_sizes.size()) { dim_sizes.push_back(all_sizes[size_idx++]); }
        }
        dimension_block_sizes[idx] = dim_sizes;
      }
    }
    // Parse coordinates (comma-separated integers, no braces)
    else if (current_line.find(',') != std::string::npos && current_line.find('{') == std::string::npos &&
             current_line.find('}') == std::string::npos) {
      std::istringstream coord_stream(current_line);
      std::string token;
      std::vector<int> coord;

      while (std::getline(coord_stream, token, ',')) {
        token.erase(0, token.find_first_not_of(" \t"));
        token.erase(token.find_last_not_of(" \t") + 1);
        if (!token.empty()) coord.push_back(std::stoi(token));
      }

      if (!coord.empty()) { coords.insert(coord); }
    }
  }

  // Create and initialize tensor
  BlockSparseTensor tensor(tensor_name, tensor_indices);
  tensor.initialize(tensor_name, tensor_indices, dimension_blocks, dimension_block_sizes, coords);

  return tensor;
}

// Parse multiple tensor files
std::vector<BlockSparseTensor> parse_tensor_files(const std::vector<std::string> &filenames) {
  std::vector<BlockSparseTensor> tensors;
  for (const auto &filename : filenames) { tensors.push_back(parse_tensor_file(filename)); }
  return tensors;
}

void remap_indices(BlockSparseTensor &tensor, const std::vector<char> &new_indices) {
  if (new_indices.size() != tensor.indices.size()) {
    throw std::invalid_argument("remap_indices: size mismatch between new indices and tensor rank");
  }
  // Build a new block_sizes map with new labels in positional order
  std::map<char, std::vector<int>> new_block_sizes;
  for (size_t i = 0; i < new_indices.size(); ++i) {
    char old_lbl = tensor.indices[i];
    char new_lbl = new_indices[i];
    auto it = tensor.block_sizes.find(old_lbl);
    if (it == tensor.block_sizes.end()) {
      throw std::runtime_error("remap_indices: missing block_sizes for old label");
    }
    new_block_sizes[new_lbl] = it->second;
  }
  tensor.indices = new_indices;
  tensor.block_sizes = std::move(new_block_sizes);
}
