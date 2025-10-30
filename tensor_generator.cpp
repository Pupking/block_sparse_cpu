#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

/**
 * @brief Structure to represent tensor metadata and coordinates
 */
struct BlockSparseTensor {
  std::string name;
  std::vector<char> indices;                    // e.g., {'i', 'j'} for tensor A(ij)
  std::vector<int> num_blocks;                  // number of blocks per dimension
  std::map<char, std::vector<int>> block_sizes; // block sizes per dimension
  std::set<std::vector<int>> coordinates;       // non-zero block coordinates
  double sparsity;                              // target sparsity ratio

  BlockSparseTensor(const std::string &n, const std::vector<char> &idx, double sp = 0.3)
      : name(n), indices(idx), sparsity(sp) {}
};

/**
 * @brief Generate random block sizes for a dimension, constraining the total size.
 * The total dimension size is steered towards mean(min, max) * num_blocks to
 * avoid excessively large dimensions from pure randomness.
 */
std::vector<int> generate_block_sizes(int num_blocks, int min_size, int max_size, std::mt19937 &rng) {
  if (num_blocks <= 0) { return {}; }

  std::vector<int> sizes(num_blocks);
  long long target_sum =
      static_cast<long long>(round((static_cast<double>(min_size + max_size) / 2.0) * num_blocks));
  long long current_sum = 0;

  for (int i = 0; i < num_blocks - 1; ++i) {
    long long remaining_sum = target_sum - current_sum;
    int remaining_blocks = num_blocks - i;

    // Determine bounds to guide the sum towards the target
    int upper_bound = static_cast<int>(remaining_sum - (long long)(remaining_blocks - 1) * min_size);
    upper_bound = std::min(max_size, upper_bound);

    int lower_bound = static_cast<int>(remaining_sum - (long long)(remaining_blocks - 1) * max_size);
    lower_bound = std::max(min_size, lower_bound);

    // If bounds are illogical, fall back to the original range
    if (lower_bound > upper_bound) {
      lower_bound = min_size;
      upper_bound = max_size;
    }

    std::uniform_int_distribution<int> dist(lower_bound, upper_bound);
    int size = dist(rng);
    sizes[i] = size;
    current_sum += size;
  }

  // Set the last block to meet the target, then clamp to the valid range.
  int last_size = static_cast<int>(target_sum - current_sum);
  sizes[num_blocks - 1] = std::clamp(last_size, min_size, max_size);

  return sizes;
}

/**
 * @brief Generate random coordinates with target sparsity
 */
void generate_random_coordinates(BlockSparseTensor &tensor, std::mt19937 &rng) {
  // Calculate total possible blocks
  size_t total_blocks = 1;
  for (int num : tensor.num_blocks) { total_blocks *= num; }

  // Calculate target number of non-zero blocks
  size_t target_nnz = std::max(1, static_cast<int>(total_blocks * tensor.sparsity));

  // Generate all possible coordinates
  std::vector<std::vector<int>> all_coords;
  std::function<void(std::vector<int> &, int)> generate_all;

  generate_all = [&](std::vector<int> &current, int dim) {
    if ((size_t)dim == tensor.indices.size()) {
      all_coords.push_back(current);
      return;
    }

    for (int i = 0; i < tensor.num_blocks[dim]; ++i) {
      current.push_back(i);
      generate_all(current, dim + 1);
      current.pop_back();
    }
  };

  std::vector<int> coord;
  generate_all(coord, 0);

  // Randomly select coordinates
  std::shuffle(all_coords.begin(), all_coords.end(), rng);

  for (size_t i = 0; i < std::min(target_nnz, all_coords.size()); ++i) { tensor.coordinates.insert(all_coords[i]); }
}

/**
 * @brief Generate compatible tensors for contraction
 */
class TensorGenerator {
private:
  std::mt19937 rng;
  int min_block_size;
  int max_block_size;

public:
  TensorGenerator(unsigned seed = std::random_device{}(), int min_size = 5, int max_size = 20)
      : rng(seed), min_block_size(min_size), max_block_size(max_size) {}

  /**
   * @brief Generate tensors for einsum operation (e.g., "ij,jk->ik")
   */
  std::vector<BlockSparseTensor> generate_einsum_tensors(const std::string &einsum_str,
                                                         const std::vector<int> &dimension_blocks,
                                                         const std::vector<double> &sparsities) {
    // Parse einsum string
    std::vector<std::string> parts;
    std::stringstream ss(einsum_str);
    std::string part;

    while (std::getline(ss, part, ',')) { parts.push_back(part); }

    // Split last part on "->"
    std::string last_part = parts.back();
    parts.pop_back();

    size_t arrow_pos = last_part.find("->");
    if (arrow_pos != std::string::npos) {
      parts.push_back(last_part.substr(0, arrow_pos));
      parts.push_back(last_part.substr(arrow_pos + 2));
    }

    // Extract unique dimensions
    std::set<char> unique_dims;
    for (const auto &p : parts) {
      for (char c : p) {
        if (c != ' ') unique_dims.insert(c);
      }
    }

    // Assign block counts to each dimension
    std::map<char, int> dim_blocks;
    std::vector<char> dims_vec(unique_dims.begin(), unique_dims.end());

    if (dimension_blocks.size() != dims_vec.size()) {
      throw std::invalid_argument("Dimension blocks count mismatch with unique dimensions");
    }

    for (size_t i = 0; i < dims_vec.size(); ++i) { dim_blocks[dims_vec[i]] = dimension_blocks[i]; }

    // Generate block sizes for each dimension
    std::map<char, std::vector<int>> global_block_sizes;
    for (char dim : dims_vec) {
      global_block_sizes[dim] = generate_block_sizes(dim_blocks[dim], min_block_size, max_block_size, rng);
    }

    // Create tensors
    std::vector<BlockSparseTensor> tensors;

    for (size_t i = 0; i < parts.size(); ++i) {
      std::string tensor_name = (i < parts.size() - 1) ? std::string(1, 'A' + i) : "Output";

      std::vector<char> tensor_indices;
      for (char c : parts[i]) {
        if (c != ' ') tensor_indices.push_back(c);
      }

      double sparsity = (i < sparsities.size()) ? sparsities[i] : (sparsities.empty() ? 0.3 : sparsities.back());
      BlockSparseTensor tensor(tensor_name, tensor_indices, sparsity);

      // Set dimensions
      for (char idx : tensor_indices) { tensor.num_blocks.push_back(dim_blocks[idx]); }

      // Set block sizes
      tensor.block_sizes = global_block_sizes;

      // Generate coordinates
      if (tensor.name != "Output") { // Do not generate coordinates for the output tensor
        generate_random_coordinates(tensor, rng);
      }

      tensors.push_back(tensor);
    }

    return tensors;
  }
};

/**
 * @brief Write tensor to .bs file format
 */
void write_tensor_file(const BlockSparseTensor &tensor, const std::string &filename) {
  std::ofstream file(filename);
  if (!file.is_open()) { throw std::runtime_error("Cannot open file for writing: " + filename); }

  file << "# -----------------" << std::endl;
  file << "# Tensor " << tensor.name << " (";
  for (size_t i = 0; i < tensor.indices.size(); ++i) {
    file << tensor.indices[i];
    if (i < tensor.indices.size() - 1) file << "";
  }
  file << ")" << std::endl;
  file << "# -----------------" << std::endl;

  // Write number of sections
  file << "# Number of sections {";
  for (size_t i = 0; i < tensor.indices.size(); ++i) {
    file << tensor.indices[i];
    if (i < tensor.indices.size() - 1) file << ", ";
  }
  file << "}" << std::endl;

  file << "{";
  for (size_t i = 0; i < tensor.num_blocks.size(); ++i) {
    file << tensor.num_blocks[i];
    if (i < tensor.num_blocks.size() - 1) file << ", ";
  }
  file << "}" << std::endl;

  // Write nnz
  file << "# Number of non-zero blocks (nnz)" << std::endl;
  file << tensor.coordinates.size() << std::endl;

  // Write block sizes
  file << "# Sizes for ";
  for (size_t i = 0; i < tensor.indices.size(); ++i) {
    char idx = tensor.indices[i];
    file << idx << " sections {";
    const auto &sizes = tensor.block_sizes.at(idx);
    for (size_t j = 0; j < sizes.size(); ++j) {
      file << sizes[j];
      if (j < sizes.size() - 1) file << ", ";
    }
    file << "}";
    if (i < tensor.indices.size() - 1) file << " and ";
  }
  file << std::endl;

  file << "{";
  for (size_t i = 0; i < tensor.indices.size(); ++i) {
    char idx = tensor.indices[i];
    const auto &sizes = tensor.block_sizes.at(idx);
    for (size_t j = 0; j < sizes.size(); ++j) {
      file << sizes[j];
      if (!(i == tensor.indices.size() - 1 && j == sizes.size() - 1)) { file << ", "; }
    }
  }
  file << "}" << std::endl;

  // Write coordinates
  file << "# Non-zero block coordinates (";
  for (size_t i = 0; i < tensor.indices.size(); ++i) {
    file << tensor.indices[i];
    if (i < tensor.indices.size() - 1) file << ", ";
  }
  file << ")" << std::endl;

  for (const auto &coord : tensor.coordinates) {
    for (size_t i = 0; i < coord.size(); ++i) {
      file << coord[i];
      if (i < coord.size() - 1) file << ", ";
    }
    file << std::endl;
  }

  file.close();
}

/**
 * @brief Print usage information
 */
void print_usage(const char *program_name) {
  std::cout << "Usage: " << program_name << " [OPTIONS]" << std::endl;
  std::cout << "Generate random block-sparse tensor coordinates for contractions" << std::endl;
  std::cout << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  -e, --einsum STR      Einsum operation (e.g., 'ij,jk->ik')" << std::endl;
  std::cout << "  -d, --dims N1,N2,...  Number of blocks per dimension" << std::endl;
  std::cout << "  -s, --sparsity S1,S2  Sparsity ratios (0.0-1.0)" << std::endl;
  std::cout << "  --min-block MIN       Minimum block size (default: 5)" << std::endl;
  std::cout << "  --max-block MAX       Maximum block size (default: 20)" << std::endl;
  std::cout << "  --seed SEED           Random seed (default: random)" << std::endl;
  std::cout << "  -o, --output PREFIX   Output file prefix (default: tensor)" << std::endl;
  std::cout << "  -h, --help            Show this help" << std::endl;
  std::cout << std::endl;
  std::cout << "Examples:" << std::endl;
  std::cout << "  " << program_name << " -e 'ij,jk->ik' -d 3,4,2 -s 0.4,0.3" << std::endl;
  std::cout << "  " << program_name << " -e 'abc,bcd->ad' -d 2,3,4,2 -s 0.5,0.4" << std::endl;
}

int main(int argc, char *argv[]) {
  std::string einsum_str = "ij,jk->ik";
  std::vector<int> dimension_blocks = {3, 4, 2};
  std::vector<double> sparsities = {0.4, 0.3};
  int min_block_size = 5;
  int max_block_size = 20;
  unsigned seed = std::random_device{}();
  std::string output_prefix = "tensor";

  // Parse command line arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      return 0;
    } else if (arg == "-e" || arg == "--einsum") {
      if (i + 1 < argc) { einsum_str = argv[++i]; }
    } else if (arg == "-d" || arg == "--dims") {
      if (i + 1 < argc) {
        dimension_blocks.clear();
        std::stringstream ss(argv[++i]);
        std::string item;
        while (std::getline(ss, item, ',')) { dimension_blocks.push_back(std::stoi(item)); }
      }
    } else if (arg == "-s" || arg == "--sparsity") {
      if (i + 1 < argc) {
        sparsities.clear();
        std::stringstream ss(argv[++i]);
        std::string item;
        while (std::getline(ss, item, ',')) { sparsities.push_back(std::stod(item)); }
      }
    } else if (arg == "--min-block") {
      if (i + 1 < argc) { min_block_size = std::stoi(argv[++i]); }
    } else if (arg == "--max-block") {
      if (i + 1 < argc) { max_block_size = std::stoi(argv[++i]); }
    } else if (arg == "--seed") {
      if (i + 1 < argc) { seed = std::stoul(argv[++i]); }
    } else if (arg == "-o" || arg == "--output") {
      if (i + 1 < argc) { output_prefix = argv[++i]; }
    }
  }

  try {
    std::cout << "Block-Sparse Tensor Generator" << std::endl;
    std::cout << "=============================" << std::endl;
    std::cout << "Einsum operation: " << einsum_str << std::endl;
    std::cout << "Dimension blocks: ";
    for (size_t i = 0; i < dimension_blocks.size(); ++i) {
      std::cout << dimension_blocks[i];
      if (i < dimension_blocks.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;

    std::cout << "Sparsities: ";
    for (size_t i = 0; i < sparsities.size(); ++i) {
      std::cout << sparsities[i];
      if (i < sparsities.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;

    std::cout << "Block size range: [" << min_block_size << ", " << max_block_size << "]" << std::endl;
    std::cout << "Random seed: " << seed << std::endl;
    std::cout << std::endl;

    // Generate tensors
    TensorGenerator generator(seed, min_block_size, max_block_size);
    auto tensors = generator.generate_einsum_tensors(einsum_str, dimension_blocks, sparsities);

    // Write input tensor files, skipping the output tensor
    for (const auto &tensor : tensors) {
      if (tensor.name == "Output") { continue; }

      std::string filename = output_prefix + "_" + tensor.name + ".bs";
      write_tensor_file(tensor, filename);

      std::cout << "Generated " << tensor.name << " (" << tensor.coordinates.size() << " non-zero blocks) -> "
                << filename << std::endl;
    }

    std::cout << std::endl << "Generation complete!" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
