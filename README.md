<p align="center">
  <a href="https://github.com/AlayaDB-AI"><img src="https://github.com/AlayaDB-AI/AlayaLite/blob/main/docs/images/banner.jpg?raw=true" width=300 alt="AlayaDB Log"></a>
</p>


<p align="center">
    <b>AlayaLite – A Fast, Flexible Vector Database for Everyone</b>. <br />
    Seamless Knowledge, Smarter Outcomes.
</p>

<div class="column" align="middle">
  <a href="https://github.com/AlayaDB-AI/AlayaLite/releases"><img height="20" src="https://img.shields.io/badge/alayalite-blue" alt="release"></a>
  <a href="https://pypi.org/project/alayalite/"><img src="https://img.shields.io/pypi/v/alayalite" alt="PyPi"></a>
  <a href="https://github.com/AlayaDB-AI/AlayaLite/blob/main/LICENSE"><img height="20" src="https://img.shields.io/badge/license-AGPL--3.0-blue.svg" alt="LICENSE"></a>
  <a href="https://codecov.io/github/AlayaDB-AI/AlayaLite"><img height="20" src="https://codecov.io/github/AlayaDB-AI/AlayaLite/graph/badge.svg?token=KA6V0DHHUU" alt="codecov"></a>
  <a href="https://github.com/AlayaDB-AI/AlayaLite/actions/workflows/code-checker.yaml"><img height="20" src="https://github.com/AlayaDB-AI/AlayaLite/actions/workflows/code-checker.yaml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://discord.gg/ReEHqSx97"><img height="20" src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://x.com/home"><img height="20" src="https://img.shields.io/badge/X-Follow-000000?logo=x&logoColor=white" alt="X"></a>
</div>


## Features

- **High Performance**: Modern vector techniques integrated into a well-designed architecture.
- **Elastic Scalability**: Seamlessly scale across multiple threads, which is optimized by C++20 coroutines.
- **Adaptive Flexibility**: Easy customization for quantization methods, metrics, and data types.
- **Two public index families**: exact `flat` and stable-id `qg`. Eligible
  sealed `qg` generations use the on-disk **LASER** implementation.
- **Native LASER active surface**: C++ builds on Linux x86_64 can also opt
  into the writable LASER active engine; this is separate from the Python
  `flat`/`qg` family names.
- **Lean distribution**: the AlayaLite 1.2.0 reference wheel is 5.9 MB.
- **Ease of Use**: [Intuitive APIs](https://github.com/AlayaDB-AI/AlayaLite/blob/main/python/README.md) in Python.

## Documentation

- [Client User Guide](https://github.com/AlayaDB-AI/AlayaLite/blob/main/docs/user/CLIENT_USER_MANUAL.md)
- [Build Guide](https://github.com/AlayaDB-AI/AlayaLite/blob/main/docs/user/BUILDING.md)
- [LASER implementation guide](https://github.com/AlayaDB-AI/AlayaLite/blob/main/docs/design/LASER.md)

## Quick Start

Get started with just one command!

```bash
pip install alayalite             # with pip
# or
uv pip install alayalite          # with uv (standalone)
uv add alayalite                  # in a uv-managed project
```



### Collection quick start

```python
import numpy as np
from alayalite import Client

client = Client()
vectors = np.random.rand(1000, 128).astype(np.float32)
queries = np.random.rand(10, 128).astype(np.float32)
collection = client.create_collection(
    "docs",
    index_type="flat",  # available on every wheel
    metric="l2",
    indexed_fields=["category"],
)
collection.insert(
    [
        (f"doc-{i}", f"Document {i}", vector, {"category": "database" if i % 2 == 0 else "other"})
        for i, vector in enumerate(vectors)
    ]
)

result = collection.hybrid_query(
    vectors=queries,
    limit=10,
    metadata_filter={"category": "database"},
    ef_search=100,
)
print(result["id"][0])
```

### Index and platform matrix

| Wheel/build | Python `flat` | Python `qg` seal | C++ writable LASER active engine |
|---|---:|---:|---:|
| Linux x86_64 | yes | LASER (`libaio` by default) | yes |
| macOS x86_64/arm64 | yes | LASER (thread pool) | no |
| Linux aarch64 | yes | unavailable; fails explicitly | no |
| Windows x64 | yes | unavailable; fails explicitly | no |

Linux aarch64 and Windows are therefore operationally flat-only. They still
validate the stable `qg` name, but sealing it raises
`CollectionNotSupportedError`; AlayaLite never silently substitutes Flat for
a platform-gated `qg` request. See the build and LASER implementation guides
above for native requirements and schema limits.

## Benchmark

The figures below are historical result snapshots, retained for research
context. Their retired Python runners and ANN-Benchmarks adapter are not part
of the current repository surface; current native targets live under
`benchmarks/`.

### Historical: in-memory graph vs. ANN-Benchmarks

The former in-memory graph path was evaluated against other vector database systems using
[ANN-Benchmark](https://github.com/erikbern/ann-benchmarks) (compile locally and
open `-march=native` in your `CMakeLists.txt` to reproduce the results).

|     ![Fashion-MNIST	784 Euclidean](https://raw.githubusercontent.com/AlayaDB-AI/AlayaLite/main/docs/images/fashion-mnist-784-euclidean.png)     |    ![Gist 960 Euclidean](https://raw.githubusercontent.com/AlayaDB-AI/AlayaLite/main/docs/images/gist-960-euclidean.png)    |
| :---------------------------------------------------------: | :-----------------------------------------------------------: |
| <div style="text-align: center;">**Fashion-MNIST	784 Euclidean**</div> | <div style="text-align: center;">**Gist 960 Euclidean**</div> |

### Historical: pre-cutover collection vs. other mainstream systems

The pre-cutover in-memory path also powered `Collection` hybrid search when metadata filters
were involved. This filtered retrieval workflow was evaluated using
[VectorDBBench](https://github.com/zilliztech/VectorDBBench) on the
**Medium Cohere** dataset (1M vectors, 768 dimensions). The following results
report QPS under 0.1% selectivity filters at concurrency 1 and 80.

![Integer filter 0.1% selectivity QPS](https://raw.githubusercontent.com/AlayaDB-AI/AlayaLite/main/docs/images/int-0p1p_qps_c1_c80.png)

![String equality filter 0.1% selectivity QPS](https://raw.githubusercontent.com/AlayaDB-AI/AlayaLite/main/docs/images/strequ-0p1p_qps_c1_c80.png)

### Historical measurement: on-disk LASER vs. other large-scale systems

For the on-disk path, this snapshot compares LASER against other disk-resident vector
systems on **DPR100M** (101M vectors × 768 dimensions, L2). Numbers are read
directly from the benchmark output — see the
[AlayaLaser paper](https://arxiv.org/abs/2602.23342) (SIGMOD 2026) for the
algorithm details.

![LASER vs other on-disk systems on DPR100M](https://raw.githubusercontent.com/AlayaDB-AI/AlayaLite/main/docs/images/laser-vs-disk-anns.png)

At Recall@10 ≈ 0.97, LASER serves about **725 QPS** — roughly 4.4× DiskANN
(165), 9.2× Qdrant (79), and 66× LanceDB (11) on this dataset, while Milvus
(3) does not reach this recall band reliably. The search-phase resident set
is **22.5 GiB**, an order of magnitude below Qdrant (309.2 GiB) and Milvus
(382.1 GiB) on the same workload.



## Contributing

We welcome contributions to AlayaLite! If you would like to contribute, please follow these steps:

1. Start by creating an issue outlining the feature or bug you plan to work on.
2. We will collaborate on the best approach to move forward based on your issue.
3. Fork the repository, implement your changes, and commit them with a clear message.
4. Push your changes to your forked repository.
5. Submit a pull request to the main repository.

Please ensure that your code follows the coding standards of the project and includes appropriate tests.

## Acknowledgements

We would like to thank all the contributors and users of AlayaLite for their support and feedback.

## Contact

If you have any questions or suggestions, please feel free to open an issue or contact us at **dev@alayadb.ai**.

For Chinese-speaking users, you can join our WeChat discussion group by scanning the QR code below:

<p align="center">
  <img src="https://raw.githubusercontent.com/AlayaDB-AI/AlayaLite/main/docs/images/wechat-group-qr.png" width="240" alt="AlayaLite WeChat discussion group QR code">
</p>


## License

[AGPL-3.0](https://github.com/AlayaDB-AI/AlayaLite/blob/main/LICENSE)
