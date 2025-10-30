#include "block_sparse_matrix.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

/**
 * @brief Parse a .bs file and populate a BlockSparseMatrix
 * @param filename Path to the .bs file
 * @return BlockSparseMatrix with parsed data
 */
BlockSparseMatrix parse_bs_file(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) { throw std::runtime_error("Could not open file: " + filename); }

  std::string line;
  std::vector<std::string> lines;

  // Read all lines
  while (std::getline(file, line)) { lines.push_back(line); }
  file.close();

  // Parse dimensions
  int num_i_blocks = 0, num_j_blocks = 0;
  std::vector<int> sizes_i, sizes_j;
  std::vector<std::pair<int, int>> coordinates;

  for (size_t i = 0; i < lines.size(); ++i) {
    const std::string &current_line = lines[i];

    // Skip comments and empty lines
    if (current_line.empty() || current_line[0] == '#') { continue; }

    // Parse dimensions line: {num_i, num_j}
    if (current_line.find('{') != std::string::npos && current_line.find('}') != std::string::npos) {
      if (num_i_blocks == 0 && num_j_blocks == 0) {
        // First occurrence is dimensions
        std::string dim_str =
          current_line.substr(current_line.find('{') + 1, current_line.find('}') - current_line.find('{') - 1);

        std::istringstream dim_stream(dim_str);
        std::string token;
        std::getline(dim_stream, token, ',');
        num_i_blocks = std::stoi(token);
        std::getline(dim_stream, token);
        num_j_blocks = std::stoi(token);

        std::cout << "Parsed dimensions: " << num_i_blocks << " x " << num_j_blocks << std::endl;
      } else {
        // Second occurrence is sizes
        std::string sizes_str =
          current_line.substr(current_line.find('{') + 1, current_line.find('}') - current_line.find('{') - 1);

        std::istringstream sizes_stream(sizes_str);
        std::string token;

        // Parse i sizes
        for (int j = 0; j < num_i_blocks; ++j) {
          if (std::getline(sizes_stream, token, ',')) { sizes_i.push_back(std::stoi(token)); }
        }

        // Parse j sizes
        for (int j = 0; j < num_j_blocks; ++j) {
          if (std::getline(sizes_stream, token, ',')) { sizes_j.push_back(std::stoi(token)); }
        }

        std::cout << "Parsed " << sizes_i.size() << " i sizes and " << sizes_j.size() << " j sizes" << std::endl;
      }
    }

    // Parse coordinates
    if (current_line.find(',') != std::string::npos && current_line.find('{') == std::string::npos &&
        current_line.find('}') == std::string::npos) {
      std::istringstream coord_stream(current_line);
      std::string token;

      if (std::getline(coord_stream, token, ',')) {
        int i = std::stoi(token);
        if (std::getline(coord_stream, token)) {
          int j = std::stoi(token);
          coordinates.emplace_back(i, j);
        }
      }
    }
  }

  // Validate parsed data
  if (num_i_blocks == 0 || num_j_blocks == 0) {
    throw std::runtime_error("Failed to parse dimensions from file: " + filename);
  }

  if (sizes_i.size() != static_cast<size_t>(num_i_blocks)) {
    throw std::runtime_error("Mismatch in i dimension sizes: expected " + std::to_string(num_i_blocks) + ", got " +
                             std::to_string(sizes_i.size()));
  }

  if (sizes_j.size() != static_cast<size_t>(num_j_blocks)) {
    throw std::runtime_error("Mismatch in j dimension sizes: expected " + std::to_string(num_j_blocks) + ", got " +
                             std::to_string(sizes_j.size()));
  }

  if (coordinates.empty()) { throw std::runtime_error("No coordinates found in file: " + filename); }

  std::cout << "Parsed " << coordinates.size() << " non-zero block coordinates" << std::endl;

  // Create and initialize the matrix
  BlockSparseMatrix matrix;
  matrix.initialize(num_i_blocks, num_j_blocks, sizes_i, sizes_j, coordinates);

  return matrix;
}

/**
 * @brief Parse multiple .bs files for matrices A, B, and optionally C
 * @param matrix_a_filename Filename for matrix A
 * @param matrix_b_filename Filename for matrix B
 * @param matrix_c_filename Optional filename for matrix C (reference)
 * @return Tuple of parsed matrices
 */
std::tuple<BlockSparseMatrix, BlockSparseMatrix, std::optional<BlockSparseMatrix>>
parse_matrices(const std::string &matrix_a_filename, const std::string &matrix_b_filename,
               const std::string &matrix_c_filename) {
  std::cout << "Parsing matrix A from: " << matrix_a_filename << std::endl;
  BlockSparseMatrix matrix_a = parse_bs_file(matrix_a_filename);

  std::cout << "Parsing matrix B from: " << matrix_b_filename << std::endl;
  BlockSparseMatrix matrix_b = parse_bs_file(matrix_b_filename);

  std::optional<BlockSparseMatrix> matrix_c;
  if (!matrix_c_filename.empty()) {
    try {
      std::cout << "Parsing matrix C from: " << matrix_c_filename << std::endl;
      matrix_c = parse_bs_file(matrix_c_filename);
    } catch (const std::exception &e) {
      std::cout << "Warning: Could not parse matrix C: " << e.what() << std::endl;
      std::cout << "Will compute C from symbolic phase" << std::endl;
    }
  }

  return {std::move(matrix_a), std::move(matrix_b), std::move(matrix_c)};
}
