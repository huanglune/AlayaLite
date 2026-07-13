# Native binding size map

Build the release Python module, then run:

```bash
python scripts/size_map/generate_size_map.py
```

Pass `--wheel dist/alayalite-....whl` after packaging to record the exact wheel
size as well. Since Gate 11 no longer links the 33 legacy `PyIndex` template
rows, `baseline.json` records the canonical pybind object/module sizes and a
demangled-symbol absence audit. The 33 rows remain only as generated canonical
identity test data. Compare with the same toolchain and build flags.
