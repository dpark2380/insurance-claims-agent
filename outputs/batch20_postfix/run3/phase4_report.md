# Phase 4 cost/latency report

n claims: 20
mean wall-clock latency/claim (end to end): 25.17s
p95 wall-clock latency/claim (end to end): 52.50s
mean API latency/claim: 20.30s
p95 API latency/claim: 43.59s
mean cost/claim (incl. extraction fallback): $0.0375
total cost: $0.7504
extraction fallback calls: 2 ($0.0054)
prompt cache tokens: 0 written, 0 read (no cache writes: cached prefix is under the model's minimum cacheable length or unmarked)
one-time startup (LoRA load + index build, excluded above): 10.36s
