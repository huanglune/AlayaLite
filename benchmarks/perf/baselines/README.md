# Hot-path performance baseline

The checked-in JSON files are historical, machine-specific captures. They
record the pre-retirement SIMD/QG/LASER benchmark surface on one host and must
not be presented as current performance or compared across machines.

Their runners were removed after the referenced QG test target and public
LASER Python builder were retired. Current native microbenchmarks are built
from `benchmarks/{simd,rabitq}/` and the top-level benchmark targets; there is
no unified baseline runner at present.
