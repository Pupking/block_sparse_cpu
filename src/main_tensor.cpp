#include "block_sparse_tensor.h"
#include "tblis_bs_contraction.h"
#include "tensor_utils.h"
#include "validation.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

inline void ensure_rank_supported(const BlockSparseTensor &A, const BlockSparseTensor &B, const EinsumOperation &op) {
  auto out_rank = op.output_indices.size();
  if (A.indices.size() > MAX_TENSOR_RANK || B.indices.size() > MAX_TENSOR_RANK || out_rank > MAX_TENSOR_RANK) {
    throw std::runtime_error("Tensor rank exceeds MAX_TENSOR_RANK fast path");
  }
}

template <typename T> struct PreparedContraction {
  std::vector<std::vector<T>> A_blocks;
  std::vector<std::vector<T>> B_blocks;
  DenseHostTensor<T> A_dense;
  DenseHostTensor<T> B_dense;
};

template <typename T>
PreparedContraction<T> prepare_contraction(const BlockSparseTensor &A, const BlockSparseTensor &B) {
  PreparedContraction<T> prep;
  prep.A_blocks = generate_block_values<T>(A, 123u);
  prep.B_blocks = generate_block_values<T>(B, 456u);
  prep.A_dense = pack_blocks_to_dense<T>(A, prep.A_blocks);
  prep.B_dense = pack_blocks_to_dense<T>(B, prep.B_blocks);
  return prep;
}

template <typename T> struct DenseRunResult {
  DenseHostTensor<T> tensor;
  float kernel_ms = 0.0f;
  long double flops = 0.0L;
};
  template <typename T>
  DenseRunResult<T> run_tblis_dense_impl(const BlockSparseTensor &A, const BlockSparseTensor &B,
                                      const EinsumOperation &operation, const PreparedContraction<T> &prep) {
    //throw std::runtime_error("TBLIS dense backend not implemented in this example.");
    const auto &Adense = prep.A_dense;
    const auto &Bdense = prep.B_dense;
    long double flop = tensor_utils::dense_flop_count(operation, operation.input_indices[0], operation.input_indices[1],
                                                    Adense.extents, Bdense.extents);
    float ms = 0.0f;
    auto tensor = run_dense_tblis<T>(operation.input_indices[0], operation.input_indices[1], operation.output_indices, Adense, Bdense, ms);
    std::cout << "kernel_ms: " << ms << ", gflops_eq: " << tensor_utils::gflops_from(flop,ms) << std::endl;
    DenseRunResult<T> out;
    out.tensor = std::move(tensor);
    out.kernel_ms = ms;
    out.flops = flop;
    return out;
  }

  template <typename T>
  BlockSparseNumericResult<T> run_tblis_blocksparse_impl(const BlockSparseTensor &A, const BlockSparseTensor &B,
                                                 const EinsumOperation &operation, const PreparedContraction<T> &prep,
                                                 bool do_verify) {
    //throw std::runtime_error("TBLIS block-sparse backend not implemented in this example.");
    ensure_rank_supported(A, B, operation);
    const auto &Ablocks = prep.A_blocks;
    const auto &Bblocks = prep.B_blocks;
    const auto &Adense = prep.A_dense;
    const auto &Bdense = prep.B_dense;
    auto res = contract_block_sparse_and_accumulate_tblis<T>(A, B, operation, Ablocks, Bblocks);
    long double flop = res.flops;
    std::cout << "kernel_ms: " << res.kernel_ms << ", gflops: " << tensor_utils::gflops_from(flop, res.kernel_ms)
              << std::endl;
    if (do_verify) {
      std::cout << "Verifying vs Dense..." << std::endl;
      float dense_ms = 0.0f;
      auto Od = run_dense_tblis<T>(operation.input_indices[0], operation.input_indices[1], operation.output_indices,
                                      Adense, Bdense, dense_ms);
      auto Ddense = pack_blocks_to_dense<T>(res.tensor, res.blocks);
      constexpr double eps = std::is_same<T, double>::value ? 1e-12 : 1e-7;
      tensor_utils::print_error_metrics<T>(Od, Ddense, eps, "verify(dense)");
    }
    return res;
  }

} // anonymous namespace

void print_usage(const char *program_name) {
  std::cout << "Usage: " << program_name << " <einsum_operation> <input_files...>" << std::endl;
  std::cout << "Multi-dimensional block-sparse tensor contraction" << std::endl;
  std::cout << std::endl;
  std::cout << "Arguments:" << std::endl;
  std::cout << "  einsum_operation     Einsum notation (e.g., 'ij,jk->ik', 'abcd,cdef->abef')" << std::endl;
  std::cout << "  input_files          Input tensor files (.bs format)" << std::endl;
  std::cout << std::endl;
  std::cout << "Examples:" << std::endl;
  std::cout << "  " << program_name << " 'ij,jk->ik' tensor_A.bs tensor_B.bs" << std::endl;
  std::cout << "  " << program_name << " 'abcd,cdef->abef' tensor_A.bs tensor_B.bs" << std::endl;
  std::cout << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  --backend=tblis-dense|tblis-blocksparse" << std::endl;
  //  std::cout << "  --backend=cutensor|dense|cutlass-grouped|cutlass-single|tblis|all" << std::endl;
  std::cout << "  --dtype=f32|f64    Numeric type (default f32)" << std::endl;
  std::cout << "  --verify           Compare results to references" << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    print_usage(argv[0]);
    return 1;
  }

  try {
    std::string dtype = "f32";
    std::string backend = "all";
    bool do_verify = false;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg.rfind("--backend=", 0) == 0) {
        backend = arg.substr(std::string("--backend=").size());
      } else if (arg.rfind("--dtype=", 0) == 0) {
        dtype = arg.substr(std::string("--dtype=").size());
      } else if (arg == "--verify") {
        do_verify = true;
      } else if (!arg.empty() && arg[0] == '-') {
        continue;
      } else {
        positional.push_back(arg);
      }
    }
    if (positional.size() != 3) {
      print_usage(argv[0]);
      return 1;
    }
    const std::string einsum_str = positional[0];
    std::vector<std::string> input_filenames{positional[1], positional[2]};

    std::cout << "Multi-Dimensional Block-Sparse Tensor Parser" << std::endl;
    std::cout << "===========================================" << std::endl;
    std::cout << "Einsum operation: " << einsum_str << std::endl;
    std::cout << "Input files: " << input_filenames[0] << ", " << input_filenames[1] << std::endl << std::endl;

    std::cout << "=== Parsing Input Tensors ===" << std::endl;
    std::vector<BlockSparseTensor> input_tensors;
    for (size_t i = 0; i < input_filenames.size(); ++i) {
      std::cout << "Parsing tensor " << static_cast<char>('A' + i) << " from: " << input_filenames[i] << std::endl;
      auto tensor = parse_tensor_file(input_filenames[i]);
      tensor.print_info();
      std::cout << std::endl;
      input_tensors.push_back(std::move(tensor));
    }

    if (input_tensors.size() != 2) {
      std::cerr << "Error: expected exactly 2 input tensors" << std::endl;
      return 1;
    }

    EinsumOperation operation;
    operation.parse(einsum_str);

    if (!operation.is_valid()) {
      std::cerr << "Error: invalid einsum operation" << std::endl;
      return 1;
    }
    if (operation.input_indices.size() != input_tensors.size()) {
      std::cerr << "Error: mismatch between einsum inputs and provided tensors" << std::endl;
      return 1;
    }

    for (size_t i = 0; i < input_tensors.size(); ++i) {
      if (input_tensors[i].indices != operation.input_indices[i]) {
        remap_indices(input_tensors[i], operation.input_indices[i]);
      }
    }

    auto &A = input_tensors[0];
    auto &B = input_tensors[1];

    auto dispatch_dtype = [&](auto tag) -> int {
      using T = decltype(tag);
      auto prep = prepare_contraction<T>(A, B);

      std::optional<DenseRunResult<T>> tblis_dense_res;
      std::optional<BlockSparseNumericResult<T>> tblis_bs_res;

      auto run_tblis_dense = [&](bool verify) {
        std::cout << "\n=== Run: TBLIS Dense ===" << std::endl;
        tblis_dense_res = run_tblis_dense_impl<T>(A, B, operation, prep);
      };

      auto run_tblis_blocksparse = [&](bool verify) {
        std::cout << "\n=== Run: TBLIS Block-Sparse ===" << std::endl;
        //std::cout << "\n Not implemented" << std::endl;
        tblis_bs_res = run_tblis_blocksparse_impl<T>(A, B, operation, prep, verify);
      };

      if (backend == "tblis-dense") {
        run_tblis_dense(do_verify);
      } else if (backend == "tblis-blocksparse") {
        run_tblis_blocksparse(do_verify);
      } else if (backend == "all") {
        run_tblis_dense(false);
        run_tblis_blocksparse(false);
      } else {
        std::cerr << "Unknown --backend value: " << backend << std::endl;
        return 1;
      }
      return 0;
    };

    int dispatch_rc = 0;
    if (dtype == "f64")
      dispatch_rc = dispatch_dtype(double{});
    else
      dispatch_rc = dispatch_dtype(float{});
    if (dispatch_rc != 0) return dispatch_rc;

    std::cout << "\n=== Summary ===" << std::endl;
    operation.print_info();
    std::cout << std::endl;
    std::cout << "Parsed " << input_tensors.size() << " input tensor(s)." << std::endl;
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Unknown error occurred" << std::endl;
    return 1;
  }
}
