# Native binding size map

Build the release Python module, then run:

```bash
python benchmarks/size_map/generate_size_map.py
```

Pass `--wheel dist/alayalite-....whl` after packaging to record the exact wheel
size as well. Since Gate 11 no longer links the 33 legacy `PyIndex` template
rows, `baseline.json` records the canonical pybind object/module sizes and a
demangled-symbol absence audit. `canonical_identity_rows` and
`legacy_dispatch_rows_linked` are both retired counters (fixed at 0): the
hnsw-keyed dispatch codegen matrix that used to populate the former is gone
too (HNSW retirement wave). Compare with the same toolchain and build flags.
