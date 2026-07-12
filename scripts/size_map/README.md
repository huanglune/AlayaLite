# Code-generation size map

Build the release Python module, then run:

```bash
python scripts/size_map/generate_size_map.py
```

Pass `--wheel dist/alayalite-....whl` after packaging to record the exact wheel
size as well. `baseline.json` records the extension and factory object sizes and
all 33 dispatch rows. Per-row values sum matching demangled text symbols; they
are attribution figures (nested/shared template symbols can overlap), not a
partition of the object file. Compare the same toolchain and build flags in
steps 4 and 11.
