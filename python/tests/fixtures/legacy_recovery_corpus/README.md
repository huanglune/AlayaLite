# Legacy PyIndex recovery corpus

These fixtures were written by commit `ab2cb0f` before the Segment migration.
They pin the legacy Graph/Space snapshot, custom WAL, RocksDB checkpoint,
`manifest.txt`, and `CURRENT` formats. Tests always copy a case before opening
it because the legacy reader publishes a `post_recovery` snapshot after replay.

`schema.json` and scalar `data.snapshot`/`raw.data` files use
`__CORPUS_ROOT__/rocksdb` as a relocation placeholder. The reader test replaces
it only in its temporary copy. Each case's `sha256.json` lists every other file
in that case; the checksum manifest excludes itself.

After building the Python extension from this checkout, regenerate with:

```sh
PYTHONPATH=python/src .venv/bin/python scripts/generate_legacy_recovery_corpus.py
```

RocksDB embeds random DB/session identities in SST and MANIFEST files. Normal
in-place regeneration preserves the checked-in checkpoint bytes so a clean
checkout regenerates byte-for-byte. Pass `--refresh-rocksdb` only when the
scalar checkpoint contents are intentionally being replaced; then review the
new checksums and run the read-back test.
