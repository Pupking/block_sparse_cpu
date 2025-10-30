# Block-Sparse Tools

Block-sparse tensor contraction benchmarks. The main CLI parses `.bs` inputs, prepares metadata, and runs block-sparse CPU backends (tblis). A companion generator produces compatible block-sparse inputs for quick experiments.

## Project Layout

```
block_sparse/
├── include/
│   ├── block_sparse_tensor.h      # tensor/einsum metadata types
│   ├── tblis_bs_contraction.h     # blockSparse tblis helpers
│   ├── tensor_layout.h            # shared coordinate helpers
│   └── validation.h               # dense reference helpers
├── src/
│   ├── main_tensor.cpp            # contraction CLI + backend dispatcher
│   ├── tblis_bs_contraction.cpp   # blockSparse contraction using tblis
│   ├── tensor_layout.cpp          # shared coordinate/flop helpers
│   ├── validation.cpp             # dense implementation for verification
│   └── bs_flops.cpp               # dense-equivalent FLOP counter CLI
├── tensor_generator.cpp           # block-sparse tensor generator CLI
├── src/
└── tblis/                       # expected tblis headers (subtree)
```

## Build

### Requirements

- CMake ≥ 3.23
- C++17 toolchain
- tblis 1.3(place headers/libs under
  `tblis/{include,lib}`)

### tblis install instructions
```bash
$ git clone https://github.com/MatthewsResearchGroup/tblis.git
$ git checkout tags/v1.3.0
$ cd tblis
$ ./configure --prefix=$PWD
$ make && make install
$ export LD_LIBRARY_PATH=/home/dsh/NCSU/tensor/block_sparse_cpu/tblis/lib/:LD_LIBRARY_PATH
```
### Configure & Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This builds three executables:

- `tensor_contraction` – einsum inspector + contraction CLI
- `tensor_generator` – emits compatible `.bs` inputs
- `bs_flops` – reports equivalent dense FLOPs for a `.bs` pair

## `tensor_contraction` CLI

```
./tensor_contraction [options] <einsum> <tensor_A.bs> <tensor_B.bs>
```

Options:

- `--backend=tblis-dense|tblis-blocksparse|all`
  - `tblis-blocksparse` – tblis block-sparse contraction
  - `tblis-dense` – dense tblis (uses packed dense buffers)
  - `all` – run all backends sequentially; combines timings and optionally
    cross-checks outputs when `--verify` is set
- `--dtype=f32|f64` – numeric type for generated block values (default `f32`)
- `--verify` – cross-check backend outputs against each other (and dense when
  available)

Example:

```bash
./build/tensor_contraction --backend=all --verify "abcd,cdef->abef" tensor_A.bs tensor_B.bs
```

The dispatcher prepares one set of block values/dense packs and reuses them
across every backend to keep timings comparable. When `--verify` is supplied,
results from each backend are compared in dense form.

## `tensor_generator`

```
./tensor_generator -e 'abcd,cdef->abef' -d 6,1,6,1,6,1 -s 0.5,0.5 --min-block 22 --max-block 22  -o tensor
```

Produces `tensor_A.bs`, `tensor_B.bs` with the following options:
- `-e <einsum>` – einsum expression
- `-d <dims>` – comma-separated list of global tensor dimensions
- `-s <sparsity>` – comma-separated list of per-tensor sparsity
- `--min-block <size>` – minimum block size
- `--max-block <size>` – maximum block size
- `-o <prefix>` – output prefix

## `bs_flops`

```
./bs_flops --einsum 'abcd,cdef->abef' tensor_A.bs tensor_B.bs
```

Reports the sparse block contraction FLOP count implied by the `.bs` pair.

## `.bs` File Format (Matrix Example)

```
# Tensor A (ij)
{3, 4}
# Block sizes for each section along i then j
{16, 8, 12, 10, 14, 6, 18}
# Non-zero block coordinates
0, 0
0, 2
1, 1
2, 3
```

Block sections are listed per-dimension, followed by the coordinates of each
non-zero block (0-based indices).

## Notes

- The contraction CLI currently supports binary einsum expressions where all
  contracted indices disappear from the output (e.g., `abcd,cdef->abef`).
- tblis is required for the build; the CLI exits if either
  dependency is missing at runtime.
- Use `--backend=all --verify` for regression sweeps across the available
  backends.
