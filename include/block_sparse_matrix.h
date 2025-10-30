#pragma once

#include <memory>
#include <optional>
#include <utility>
#include <vector>

// Forward declarations for CUDA types
struct cudaDeviceProp;

/**
 * @brief Structure representing a block-sparse matrix
 *
 * This class manages the memory and metadata for block-sparse matrices
 * on both CPU and GPU, following the format specified in the plan.
 */
class BlockSparseMatrix {
public:
  // Matrix dimensions
  std::pair<int, int> dim_sections; // {num_i_blocks, num_j_blocks}

  // Block sizes for each dimension
  struct DimExtents {
    std::vector<int> sizes_i; // Sizes of blocks along i dimension
    std::vector<int> sizes_j; // Sizes of blocks along j dimension
  } dim_extents;

  // Non-zero block coordinates
  std::vector<std::pair<int, int>> coordinates;

  // GPU memory management
  float *d_block_data;      // Contiguous GPU buffer for all blocks
  size_t *d_block_pointers; // GPU pointers to start of each block
  size_t total_gpu_memory;  // Total GPU memory allocated

  // Host memory for verification
  std::vector<float> h_block_data;

  // Constructor
  BlockSparseMatrix();

  // Destructor
  ~BlockSparseMatrix();

  // Copy constructor (disabled for GPU memory management)
  BlockSparseMatrix(const BlockSparseMatrix &) = delete;

  // Assignment operator (disabled for GPU memory management)
  BlockSparseMatrix &operator=(const BlockSparseMatrix &) = delete;

  // Move constructor
  BlockSparseMatrix(BlockSparseMatrix &&other) noexcept;

  // Move assignment operator
  BlockSparseMatrix &operator=(BlockSparseMatrix &&other) noexcept;

  /**
   * @brief Initialize the matrix with metadata
   * @param num_i_blocks Number of blocks along i dimension
   * @param num_j_blocks Number of blocks along j dimension
   * @param sizes_i Block sizes along i dimension
   * @param sizes_j Block sizes along j dimension
   * @param coords Non-zero block coordinates
   */
  void initialize(int num_i_blocks, int num_j_blocks, const std::vector<int> &sizes_i, const std::vector<int> &sizes_j,
                  const std::vector<std::pair<int, int>> &coords);

  /**
   * @brief Allocate GPU memory for the matrix
   */
  void allocate_gpu_memory();

  /**
   * @brief Generate random values for all blocks
   */
  void generate_random_values();

  /**
   * @brief Copy data from host to GPU
   */
  void copy_to_gpu();

  /**
   * @brief Copy data from GPU to host
   */
  void copy_from_gpu();

  /**
   * @brief Get the total number of non-zero blocks
   */
  size_t get_nnz() const { return coordinates.size(); }

  /**
   * @brief Get the total matrix dimensions
   */
  std::pair<int, int> get_total_dimensions() const;

  /**
   * @brief Get the size of a specific block
   */
  std::pair<int, int> get_block_size(int i, int j) const;

  /**
   * @brief Get the offset of a block in the contiguous buffer
   */
  size_t get_block_offset(int i, int j) const;

  /**
   * @brief Convert to dense matrix for verification
   */
  std::vector<float> to_dense() const;

  /**
   * @brief Print matrix information for debugging
   */
  void print_info() const;

  // Friend functions for external operations
  friend BlockSparseMatrix parse_bs_file(const std::string &filename);
  friend std::tuple<BlockSparseMatrix, BlockSparseMatrix, std::optional<BlockSparseMatrix>>
  parse_matrices(const std::string &matrix_a_filename, const std::string &matrix_b_filename,
                 const std::string &matrix_c_filename);
  friend BlockSparseMatrix block_sparse_gemm(const BlockSparseMatrix &A, const BlockSparseMatrix &B);
  friend bool verify_block_sparse_gemm(const BlockSparseMatrix &A, const BlockSparseMatrix &B,
                                       const BlockSparseMatrix &C_result);

private:
  /**
   * @brief Calculate the total memory needed for all blocks
   */
  size_t calculate_total_memory() const;

  /**
   * @brief Find the index of a coordinate in the coordinates vector
   */
  int find_coordinate_index(int i, int j) const;
};

// External function declarations
BlockSparseMatrix parse_bs_file(const std::string &filename);
std::tuple<BlockSparseMatrix, BlockSparseMatrix, std::optional<BlockSparseMatrix>>
parse_matrices(const std::string &matrix_a_filename, const std::string &matrix_b_filename,
               const std::string &matrix_c_filename);
BlockSparseMatrix block_sparse_gemm(const BlockSparseMatrix &A, const BlockSparseMatrix &B);
bool verify_block_sparse_gemm(const BlockSparseMatrix &A, const BlockSparseMatrix &B,
                              const BlockSparseMatrix &C_result);
