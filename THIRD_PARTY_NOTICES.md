<!--
SPDX-FileCopyrightText: 2025 AlayaDB.AI
SPDX-License-Identifier: AGPL-3.0-only
-->

# Third-Party Notices

AlayaLite incorporates or adapts code from the projects below. The corresponding
license texts are distributed in `LICENSES/`. Local changes do not remove the
upstream notices or licenses. Commit identifiers are the fixed snapshots used
for the 2026-07-12 provenance audit.

## VectorDB-NTU RaBitQ-Library

- Upstream: https://github.com/VectorDB-NTU/RaBitQ-Library
- Compared commit: `5ea4df06b8dc5de3889f16084f93544f55c77212`
- License: Apache-2.0
- Local paths: `include/utils/rabitq_utils/`, `include/simd/fht.ipp`,
  `include/index/graph/laser/qg/{qg,qg_builder}.hpp`,
  `include/index/graph/laser/quantization/{fastscan_impl,rabitq}.hpp`, and
  `include/index/graph/laser/utils/{array,buffer,io,memory,rotator,stopw,tools}.hpp`
- Modifications: namespaces, include paths, APIs, platform branches, memory
  management, quantization layouts, and LASER integration were adapted; some
  implementations were substantially extended or trimmed.

## Meta Faiss

- Upstream: https://github.com/facebookresearch/faiss
- Compared commit: `63378f0`
- License: MIT
- Local path: `include/index/graph/laser/quantization/fastscan_impl.hpp`
- Modifications: FastScan packing was adapted to LASER's fixed 32-code binary
  layout and RaBitQ lookup-table pipeline.

## Microsoft DiskANN

- Upstream: https://github.com/microsoft/DiskANN
- Compared commit: `78256bbab4685e1774e78d331e081a153be26823` (`cpp_main`)
- License: MIT
- Local paths: `include/index/graph/laser/utils/{aligned_file_reader,
  aligned_file_reader_factory,concurrent_queue}.hpp`
- Modifications: C++ reader sources were consolidated into headers and adapted
  for backend-neutral events, error handling, libaio portability, and AlayaLite
  naming. The former IOCP-derived backend has been removed.
  `threadpool_file_reader.hpp` is an Alaya implementation of the
  DiskANN-lineage reader interface, not a copied DiskANN backend.

## Yahoo Japan NGT

- Upstream: https://github.com/yahoojapan/NGT
- Compared commit: `d46c0e46d46cdb698ba10884323e90c2901b3f57`
- License: Apache-2.0
- Local path: `include/third_party/ngt/hashset.hpp`
- Modifications: the hash-based boolean set was adapted to fixed local ID types,
  aligned vector storage, local sentinels, and copy/move support.

## Ant Group vsag

- Upstream: https://github.com/antgroup/vsag
- Compared commit: `3f8f254cf01c814a8c2593e814d2ec7b906217bc`
- License: Apache-2.0
- Local path: `include/index/graph/laser/utils/pca_transform.hpp`
- Modifications: the PCA pipeline was adapted to local containers and Eigen and
  extended with persistence and inverse transformation.

## FALCONN FFHT

- Upstream: https://github.com/FALCONN-LIB/FALCONN
- Compared commit: not pinned by the 2026-07-12 audit
- License: MIT
- Local path: `include/third_party/ffht/fht_avx.hpp`
- Modifications: vendored fast Fourier Hadamard transform implementation.

## JFrog Conan CMake Dependency Provider

- Upstream: https://github.com/conan-io/cmake-conan
- Compared commit: not pinned by the 2026-07-12 audit
- License: MIT
- Local path: `cmake/vendor/conan_provider.cmake`
- Modifications: vendored dependency-provider integration.

This notice is an engineering provenance record, not legal advice. In
particular, the final license expression and lineage characterization for
`fastscan_impl.hpp` awaits legal determination.
