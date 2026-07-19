# Vamana build-primitive tests

Standalone Vamana index and Segment engines are retired. The remaining
Vamana code is a build primitive used to produce LASER topology, so this
directory tests that retained boundary:

- `vamana_build_dispatch_test` validates build dispatch and parameter gates.
- `vamana_reader_test` validates the DiskANN-compatible graph codec.
- `frozen_graph_snapshot_test` validates topology export consumed by LASER.
- `test_vamana_alignment` is a manual comparison harness and is built but not
  registered with CTest because it requires external datasets and a separately
  built DiskANN reference.

The JSON files under `fixtures/` are historical alignment captures. They are
not a current standalone Vamana product matrix.

Build the maintained targets with the Release preset:

```bash
cmake --build build/Release --target \
  vamana_build_dispatch_test vamana_reader_test frozen_graph_snapshot_test \
  test_vamana_alignment
```

Run the registered tests with:

```bash
ctest --test-dir build/Release --output-on-failure -R 'vamana|frozen_graph_snapshot'
```

`test_vamana_alignment --help` lists the external paths required for a manual
run.
