** (Deep 1M, R=64, L=128)**:

| 模式                        | Buffer         | Recall@10 | QPS       |
| --------------------------- | -------------- | --------- | --------- |
| Disk-only sequential        | 20% (130 MB)   | 99.99%    | 107       |
| Disk-only pipelined (pw=64) | 20%            | 99.99%    | 145       |
| PQ (honest, M=8, rerank×4)  | 20%            | 96.35%    | 416       |
| Disk-only sequential        | 40% (260 MB)   | 99.99%    | 127       |
| Disk-only pipelined (pw=64) | 40%            | 99.99%    | 231       |
| PQ (honest, M=8, rerank×4)  | 40%            | 96.35%    | 593       |
| Disk-only sequential        | 100%+ (651 MB) | 99.99%    | 1,992     |
| Disk-only pipelined (pw=64) | 100%+          | 99.99%    | 1,517     |
| PQ (honest, M=8, rerank×4)  | 100%+          | 96.35%    | 9,330     |
| **Yi coupled (reference)**  | **20%**        | **-**     | **1,173** |
