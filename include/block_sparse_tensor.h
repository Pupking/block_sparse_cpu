// Minimal BlockSparseTensor interfaces for input parsing and inspection
#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

// Represents a block-sparse tensor's structure (no compute, no CUDA)
class BlockSparseTensor {
public:
  std::string name;
  std::vector<char> indices;                    // e.g., {'i','j'}
  std::vector<int> num_blocks;                  // number of blocks per dimension
  std::map<char, std::vector<int>> block_sizes; // per-dimension block sizes
  std::set<std::vector<int>> coordinates;       // non-zero block coordinates

  BlockSparseTensor();
  BlockSparseTensor(const std::string &n, const std::vector<char> &idx);

  void initialize(const std::string &tensor_name, const std::vector<char> &tensor_indices,
                  const std::vector<int> &dimension_blocks,
                  const std::map<char, std::vector<int>> &dimension_block_sizes,
                  const std::set<std::vector<int>> &coords);

  std::vector<int> get_total_dimensions() const;
  std::vector<int> get_block_size(const std::vector<int> &coord) const;
  bool has_coordinate(const std::vector<int> &coord) const;
  int get_dimension_index(char label) const;
  void print_info() const;
};

// Einsum description (parsing + validation only)
struct EinsumOperation {
  std::string operation_string;
  std::vector<std::vector<char>> input_indices; // per-input indices
  std::vector<char> output_indices;             // output indices
  std::vector<char> contraction_indices;        // indices summed over

  void parse(const std::string &einsum_str);
  std::set<char> get_unique_indices() const;
  bool is_valid() const; // generic validation of op shape
  void print_info() const;
};

// Parsing helpers for .bs files
BlockSparseTensor parse_tensor_file(const std::string &filename);
std::vector<BlockSparseTensor> parse_tensor_files(const std::vector<std::string> &filenames);

// Remap tensor index labels (per-dimension) to new labels while preserving order.
// Also remaps block_sizes keys to match the new labels.
void remap_indices(BlockSparseTensor &tensor, const std::vector<char> &new_indices);
